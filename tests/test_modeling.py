import tempfile
import unittest
from pathlib import Path

import torch

from iclr import modeling
from iclr.smoke import create_tiny_model
import composition_core as core
import composition_eval as legacy


class ModelingTest(unittest.TestCase):
    def test_atomic_initialization_preserves_tied_embeddings(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            base = create_tiny_model(Path(directory) / "tiny")
            model, tokenizer, ids = modeling.load(base, initialize=True, train=True, device="cpu",
                                                  lora_dropout=0, gradient_checkpointing=False)
            self.assertTrue(model.config.tie_word_embeddings)
            self.assertEqual(model.get_input_embeddings().weight.data_ptr(),
                             model.get_output_embeddings().weight.data_ptr())
            row = {"p": 7, "depth": 1, "start": [1, 2, 3], "witness": ["SH1"]}
            row["target"] = list(core.trajectory(row["start"], row["witness"], row["p"])[-1])
            example = modeling.encode(tokenizer, {"prompt": core.plan_prompt(row),
                                                  "answer": core.program_answer(row["witness"])})
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = modeling.ce_sum(model, modeling.collate(tokenizer, [example], "cpu")) / example["target_tokens"]
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                before = modeling.program_scores(model, tokenizer, ids, row)
            model = model.merge_and_unload()
            self.assertTrue(model.config.tie_word_embeddings)
            self.assertEqual(model.get_input_embeddings().weight.data_ptr(),
                             model.get_output_embeddings().weight.data_ptr())
            saved = Path(directory) / "atomic"
            model.save_pretrained(saved)
            tokenizer.save_pretrained(saved)
            restored, tokenizer, ids = modeling.load(saved, device="cpu")
            self.assertTrue(restored.config.tie_word_embeddings)
            with torch.inference_mode():
                after = modeling.program_scores(restored, tokenizer, ids, row)
            torch.testing.assert_close(before, after, rtol=1e-6, atol=3e-6)

    def test_token_budget_keeps_the_incomplete_last_update(self):
        from iclr.train import epoch_groups, token_matched_groups
        pool = [{'labels': [-100, 1, 2, 3, 4], 'target_tokens': 4},
                {'labels': [-100] + list(range(8)), 'target_tokens': 8}]
        original = [list(row['labels']) for row in pool]
        reference = epoch_groups([pool[0], pool[1], pool[0], pool[1], pool[0]], 3, 0)
        self.assertEqual([len(group) for group in reference], [3, 2])
        budgets = [sum(item['target_tokens'] for item in group) for group in reference]
        matched = token_matched_groups(pool, budgets, 0)
        self.assertEqual([sum(item['target_tokens'] for item in group) for group in matched], budgets)
        self.assertEqual([sum(sum(label != -100 for label in item['labels']) for item in group)
                          for group in matched], budgets)
        self.assertEqual([row['labels'] for row in pool], original)

    def test_scorer_gradients_padding_and_accumulation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            base = create_tiny_model(Path(directory) / "tiny")
            model, tokenizer, token_ids = modeling.load(base, initialize=True, device="cpu")
            row = {"task_id": "probe", "task_fingerprint": "probe", "split": "dev_A", "depth": 3,
                   "p": 7, "start": [1, 2, 3], "witness": ["SH1", "SC2", "REV"]}
            row["target"] = list(core.trajectory(row["start"], row["witness"], row["p"])[-1])
            scores = modeling.program_scores(model, tokenizer, token_ids, row, batch_size=8)
            _, ranking = legacy.score_task(model, tokenizer, token_ids, row, batch_size=3)
            expected = {tuple(item["program"]): item["score"] for item in ranking["ranking"]}
            torch.testing.assert_close(scores.detach(), torch.tensor([expected[p] for p in core.enumerate_programs(3)]),
                                       rtol=1e-6, atol=3e-6)
            scores_one = modeling.program_scores(model, tokenizer, token_ids, row, batch_size=1)
            torch.testing.assert_close(scores, scores_one, rtol=1e-6, atol=3e-6)
            programs = core.enumerate_programs(3)
            mask = torch.tensor([core.verify_program(row["start"], row["target"], p, row["p"]) for p in programs])
            witness = programs.index(tuple(row["witness"]))
            loss = modeling.set_loss(scores, mask, witness, "set_mass")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
            toy = torch.tensor([.2, -.7, 1.1], requires_grad=True)
            all_correct = modeling.set_loss(toy, torch.ones(3, dtype=torch.bool), 0, "set_mass")
            self.assertEqual(float(all_correct), 0.)
            all_correct.backward()
            torch.testing.assert_close(toy.grad, torch.zeros_like(toy))
            single = torch.tensor([False, True, False])
            torch.testing.assert_close(modeling.set_loss(toy, single, 1, "set_mass"),
                                       modeling.set_loss(toy, single, 1, "single_norm"))

            examples = [modeling.encode(tokenizer, {"prompt": core.plan_prompt(row), "answer": core.program_answer(row["witness"])}),
                        modeling.encode(tokenizer, {"prompt": "RESULT:", "answer": "[1,2,3]"})]
            tokens = sum(item["target_tokens"] for item in examples)
            model.zero_grad(set_to_none=True)
            full = modeling.ce_sum(model, modeling.collate(tokenizer, examples, "cpu")) / tokens
            full.backward()
            full_gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            separate = []
            for example in examples:
                term = modeling.ce_sum(model, modeling.collate(tokenizer, [example], "cpu")) / tokens
                separate.append(term.detach())
                term.backward()
            torch.testing.assert_close(full.detach(), sum(separate), rtol=1e-6, atol=1e-6)
            for name, parameter in model.named_parameters():
                if name in full_gradients:
                    torch.testing.assert_close(parameter.grad, full_gradients[name], rtol=3e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
