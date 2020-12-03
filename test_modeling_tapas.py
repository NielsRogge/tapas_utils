def prepare_tapas_inputs_for_inference():
    # Here we've prepared the following table-question pair to test TAPAS inference on:
    # data = {'Footballer': ["Lionel Messi", "Cristiano Ronaldo"], 
    #         'Age': ["33", "35"],
    # }
    # queries = "Which footballer is 33 years old?"
    # table = pd.DataFrame.from_dict(data) 
    
    input_ids = torch.tensor([[101, 2029, 4362, 2003, 3943, 2086, 2214, 1029, 102, 4362,
          2287, 14377, 6752, 2072, 3943, 13675, 2923, 15668, 8923,  2080, 3486]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    token_type_ids = torch.tensor([0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 0, 0, 0],
                                    [1, 2, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 0, 0, 0, 0],
                                    [1, 1, 1, 0, 0, 0, 0],
                                    [1, 1, 1, 0, 0, 0, 0],
                                    [1, 2, 1, 0, 1, 2, 1],
                                    [1, 1, 2, 0, 0, 0, 0],
                                    [1, 1, 2, 0, 0, 0, 0],
                                    [1, 1, 2, 0, 0, 0, 0],
                                    [1, 1, 2, 0, 0, 0, 0],
                                    [1, 1, 2, 0, 0, 0, 0],
                                    [1, 2, 2, 0, 2, 1, 2]]])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    }


@require_torch
class TapasModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        #note that google/tapas-base should correspond to tapas_inter_masklm_base_reset
        model = TapasModel.from_pretrained("google/tapas-base")

        inputs = prepare_tapas_inputs_for_inference()
        outputs = model(**inputs)
        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]]
        )

        self.assertTrue(torch.allclose(outputs.sequence_output[:, :3, :3], expected_slice, atol=1e-4))
    
    @slow
    def test_inference_masked_lm(self):
        model = TapasForMaskedLM.from_pretrained("google/tapas-base")

        inputs = prepare_tapas_inputs_for_inference()
        outputs = model(**inputs)
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
        # note that google/tapas-base-finetuned-wtq should correspond to tapas_wtq_wikisql_sqa_inter_masklm_base_reset
        model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-sqa")

        inputs = prepare_tapas_inputs_for_inference()
        outputs = model(**inputs)
        # test the logits
        logits = outputs.logits
        expected_shape = torch.Size((1, 21))
        self.assertEqual(logits.shape, expected_shape)
        expected_tensor = torch.tensor([[-9997.22461, -9997.22461, -9997.22461, -9997.22461, -9997.22461,
        -9997.22461, -9997.22461, -9997.22461, -9997.22461, -16.2628059, 
        -10004.082, 15.4330549, 15.4330549, 15.4330549, -9990.42,
        -16.3270779, -16.3270779, -16.3270779, -16.3270779, -16.3270779, -10004.8506]])

        self.assertTrue(torch.allclose(logits, expected_tensor, atol=1e-4))

    @slow
    def test_inference_question_answering_head_weak_supervision(self):
        # note that google/tapas-base-finetuned-wtq should correspond to tapas_wtq_wikisql_sqa_inter_masklm_base_reset
        model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

        inputs = prepare_tapas_inputs_for_inference()
        outputs = model(**inputs)
        # test the logits
        logits = outputs.logits
        expected_shape = torch.Size((1, 21))
        self.assertEqual(logits.shape, expected_shape)
        expected_slice = torch.tensor([[-10096.3633, -10096.3633, -10096.3633, -10096.3633, -10096.3633,
        -10096.3633, -10096.3633, -10096.3633, -10096.3633, -180.192322,
        -10080.2305, 157.994827, 157.994827, 157.994827, -10031.3721,
        -142.52597, -142.52597, -142.52597, -142.52597, -142.52597, -10065.7256]])  # ok 

        self.assertTrue(torch.allclose(logits, expected_slice, atol=1e-4))

        # test the aggregation logits
        logits_aggregation = outputs.logits_aggregation
        expected_shape = torch.Size((1, 4))
        self.assertEqual(logits_aggregation.shape, expected_shape)
        expected_tensor = torch.tensor([[18.8111877 -9.91616917 -6.3120923 -2.97642279]]) # ok

        self.assertTrue(torch.allclose(logits_aggregation, expected_tensor, atol=1e-4))

    @slow
    def test_inference_question_answering_head_strong_supervision(self):
        # note that google/tapas-base-finetuned-wikisql-supervised should correspond to tapas_wikisql_sqa_inter_masklm_base_reset
        model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wikisql-supervised")

        inputs = prepare_tapas_inputs_for_inference()
        outputs = model(**inputs)
        # test the logits
        logits = outputs.logits
        expected_shape = torch.Size((1, 21))
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
        # note that google/tapas-base-finetuned-tabfact should correspond to tapas_tabfact_inter_masklm_base_reset
        model = TapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact")

        inputs = prepare_tapas_inputs_for_inference()
        outputs = model(*inputs)

        # test the classification logits
        logits = outputs.logits
        expected_shape = torch.Size((1, 2))
        self.assertEqual(logits.shape, expected_shape)
        expected_tensor = torch.tensor([[0.795137286 9.5572]]) # ok. Note that the PyTorch model outputs [[0.8057, 9.5281]]

        self.assertTrue(torch.allclose(outputs.logits, expected_tensor, atol=1e-4))