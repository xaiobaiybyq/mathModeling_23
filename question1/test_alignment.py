import unittest
import numpy as np
from alignment import ctc_viterbi, pool_intervals, complement_intervals, tokenize_words
from read_features import collate

class AlignmentTests(unittest.TestCase):
    def test_repeated_character_requires_blank(self):
        emissions = np.full((7, 3), -20.0)
        for t, label in enumerate([0, 1, 1, 0, 1, 2, 0]):
            emissions[t, label] = 0
        _, spans, _ = ctc_viterbi(emissions, [1, 1, 2])
        np.testing.assert_array_equal(spans, [[1, 3], [4, 5], [5, 6]])

    def test_impossible_path(self):
        with self.assertRaises(ValueError):
            ctc_viterbi(np.zeros((2, 2)), [1, 1])

    def test_weighted_pool_excludes_invalid_and_handles_gap(self):
        x, valid, mapping = pool_intervals(np.array([[0.5, 2.0], [3, 4]]), np.array([[0,1],[1,2],[2,3]]), np.array([[2.],[4.],[99.]]), np.array([True,True,False]))
        self.assertAlmostEqual(x[0,0], 10/3, places=5)
        self.assertEqual(valid.tolist(), [True, False])
        self.assertEqual(mapping, [[0,1], []])
        self.assertTrue(np.all(x[1] == 0))

    def test_source_offsets_and_gaps(self):
        text = "We're high-end, really!"
        words, offsets = tokenize_words(text)
        self.assertEqual(words, [text[a:b] for a,b in offsets])
        np.testing.assert_allclose(complement_intervals([[.2,.5],[.8,1]], 1.2), [[0,.2],[.5,.8],[1,1.2]])

    def test_number_tokens_keep_source_grouping(self):
        words, offsets = tokenize_words('In 2008, 1,500 people attended the 10th event.')
        self.assertIn('1,500', words)
        self.assertIn('10th', words)

    def test_batch_padding_preserves_sequences_above_fifty(self):
        samples=[]
        for n in [3,65]:
            sample={'valid_length':np.array(n),'word_intervals':np.ones((n,2))}
            for name in ['text','audio','vision']:
                sample[name]=np.ones((n,2))
            for name in ['text_mask','audio_mask','vision_mask','alignment_mask']:
                sample[name]=np.ones(n,bool)
            sample['alignment_review_mask']=np.zeros(n,bool)
            samples.append(sample)
        batch=collate(samples)
        self.assertEqual(batch['text'].shape,(2,65,2))
        self.assertTrue(batch['padding_mask'][0,3:].all())
        self.assertFalse(batch['audio_mask'][0,3:].any())
        self.assertTrue((batch['word_intervals'][0,3:]==-1).all())
        self.assertFalse(batch['padding_mask'][1].any())

if __name__ == '__main__':
    unittest.main()
