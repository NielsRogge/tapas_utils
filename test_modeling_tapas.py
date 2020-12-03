@require_torch
class TapasModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = TapasModel.from_pretrained("google/tapas-base")

        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        outputs = model(input_ids)
        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]]
        )

        self.assertTrue(torch.allclose(outputs.sequence_output[:, :3, :3], expected_slice, atol=1e-4))
    
    @slow
    def test_inference_masked_lm(self):
        model = TapasForMaskedLM.from_pretrained("google/tapas-base")

        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        outputs = model(input_ids)
        expected_shape = torch.Size((1, 11, 30522))
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[33.8802, -4.3103, 22.7761], [4.6539, -2.8098, 13.6253], [1.8228, -3.6898, 8.8600]]]
        )

        self.assertTrue(torch.allclose(outputs.logits[:, :3, :3], expected_slice, atol=1e-4))

    # TapasForQuestionAnswering has 3 possible ways of being fine-tuned:
    # - conversational set-up (SQA)
    # - weak supervision for aggregation (WTQ, WikiSQL)
    # - strong supervision for aggregation (WikiSQL-supervised)
    # We test all of them:
    @slow
    def test_inference_question_answering_head_conversational(self):
        model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-sqa")

        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        outputs = model(input_ids)
        # test the logits
        logits = outputs.logits
        expected_shape = torch.Size((1, 11))
        self.assertEqual(logits.shape, expected_shape)
        expected_tensor = torch.tensor([[-0.9469, 0.3913, 0.5118]])
    
    @slow
    def test_inference_question_answering_head_weak_supervision(self):
        model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        outputs = model(input_ids)
        # test the logits
        logits = outputs.logits
        expected_shape = torch.Size((1, 11))
        self.assertEqual(logits.shape, expected_shape)
        expected_tensor = torch.tensor([[-0.9469, 0.3913, 0.5118]])

        # test the aggregation logits
        logits_aggregation = outputs.logits_aggregation
        expected_shape = torch.Size((1, 4))
        self.assertEqual(logits_aggregation.shape, expected_shape)
        expected_tensor = torch.tensor([[-0.9469, 0.3913, 0.5118]])

        self.assertTrue(torch.allclose(output, expected_tensor, atol=1e-4))

    @slow
    def test_inference_question_answering_head_strong_supervision(self):
        model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wikisql-supervised")

        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        outputs = model(input_ids)
        # test the logits
        logits = outputs.logits
        expected_shape = torch.Size((1, 11))
        self.assertEqual(logits.shape, expected_shape)
        expected_tensor = torch.tensor([[-0.9469, 0.3913, 0.5118]])

        # test the aggregation logits
        logits_aggregation = outputs.logits_aggregation
        expected_shape = torch.Size((1, 4))
        self.assertEqual(logits_aggregation.shape, expected_shape)
        expected_tensor = torch.tensor([[-0.9469, 0.3913, 0.5118]])

        self.assertTrue(torch.allclose(output, expected_tensor, atol=1e-4))
    
    @slow
    def test_inference_classification_head(self):
        model = TapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact")

        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        outputs = model(input_ids)
        expected_shape = torch.Size((1, 2))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_tensor = torch.tensor([[-0.9469, 0.3913, 0.5118]])

        self.assertTrue(torch.allclose(outputs.logits, expected_tensor, atol=1e-4))