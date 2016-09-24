'''
Unit Tests for fftmatch.naive_string_match_index
'''
import unittest
import fftmatch
import numpy as np
import functools
import boyermoore
import cvmatch

def format_error_message(function_name):
    return "failed on function {}".format(function_name)

def string_match_decorator(string_matching_algorithms):
    """
    Applies the Unit Test function on all of the string_matching_algorithms
    in the string_matching_algorithms list
    """
    def inner_decorator(function):
        @functools.wraps(function)
        def inner_function(self):
            '''Class method to run string matching algorithms'''
            for sma in string_matching_algorithms:
                #try:
                function(self, sma)
                #except Exception as e:
                #    raise type(e)(e.message + '\nCaused by {}'.format(sma))
        return inner_function
    return inner_decorator

#algorithms that match a single genome to a single substring
oned_string_matching_algorithms = [fftmatch.naive_string_match_index,
                              fftmatch.fft_match_index_n_log_n,
                              fftmatch.fft_match_index_n_log_m,
                              boyermoore.boyer_moore_match_index]

#algorithms that match multiple genomes to a single substring
twod_string_matching_algorithms = [fftmatch.fft_match_index_n_sq_log_n,
                                   fftmatch.fft_match_index_n_sq_log_n_naive,
                                   fftmatch.fft_match_index_n_sq_log_m_naive,
                                   #fftmatch.fft_match_index_n_sq_log_m,
                                   cvmatch.cv_match_index,
                                   cvmatch.cv_match_index_chunk]

def ndarrays_equal(arr1, arr2):
    """
    Compares two ndarrays for equality.  This is a bit tricky for arrays
    that are not uniform in size, since arr1 == arr2 produces false if there
    are any empty arrays inside any of the rows.
    """
    if arr1.shape != arr2.shape:
        return False

    for i in range(arr1.shape[0]):
        if arr1[i].shape == 0 and arr1[i].shape == 0:
            continue
        elif not (arr1[i] == arr2[i]).all():
            #if a row in arr1 isn't equal to the row in arr2, return False
            return False

    return True

class FFTStringMatchTestRig(unittest.TestCase):
    @string_match_decorator(oned_string_matching_algorithms)
    def test_single_char_single_occurrence(self, func):
        text = "AGCT"
        patterns = ["A", "G", "C", "T"]
        expected_outputs = [[0], [1], [2], [3]]

        for pattern, expected_output in zip(patterns, expected_outputs):
            output = expected_output
            self.assertTrue((func(text=text,pattern=pattern) == output).all(),
                msg=format_error_message(func))


    @string_match_decorator(oned_string_matching_algorithms)
    def test_single_char_multiple_occurrences(self, func):
        text = "AGTC" * 3
        patterns = ["A", "G", "T", "C"]
        expected_outputs = [[0,4,8], [1,5,9], [2,6,10], [3,7,11]]

        for pattern, expected_output in zip(patterns, expected_outputs):
            output = np.array(expected_output)
            self.assertTrue((func(text=text,pattern=pattern) == output).all(),
                msg=format_error_message(func))

    @string_match_decorator(oned_string_matching_algorithms)
    def test_multi_char_single_occurrence(self, func):
        text = "AGTC"
        patterns = ["AG", "GT", "TC"]
        expected_outputs = [[0], [1], [2]]

        for pattern, expected_output in zip(patterns, expected_outputs):
            output = expected_output
            self.assertTrue(func(text=text,pattern=pattern) == output,
                msg=format_error_message(func))

    @string_match_decorator(oned_string_matching_algorithms)
    def test_single_char_no_occurrence(self, func):
        text = "AGTAGTTATATGGGATATAT"
        pattern = "C"
        self.assertTrue(len(func(text=text,pattern=pattern))==0,
                msg=format_error_message(func))

    @string_match_decorator(oned_string_matching_algorithms)
    def test_multi_char_no_occurrence(self, func):
        text = "AGATATAATTATATACCATATTACCACACA"
        pattern = "CG"
        self.assertTrue(len(func(text=text,pattern=pattern))==0,
                msg=format_error_message(func))

    @string_match_decorator(oned_string_matching_algorithms)
    def test_multi_char_single_occurrence(self, func):
        text = "AGCTACCGCATTAGG"
        patterns = [text[i:i+3] for i in range(len(text)-3+1)]

        for index, pattern in enumerate(patterns):
            self.assertTrue(
                (func(text=text,pattern=pattern)==np.array([index])).all(),
                msg=format_error_message(func)
            )
        self.assertTrue(func(text="AAAGC", pattern="GC")==np.array([3]))

    @string_match_decorator(oned_string_matching_algorithms)
    def test_multi_char_multiple_occurrence(self, func):
        text = "AGCTAACCCACGGTCAA"
        patterns = [text[i:i+3] for i in range(len(text)-3+1)]
        half_len = len(text)
        text *= 2

        for index, pattern in enumerate(patterns):
            self.assertTrue(
                (func(text=text,pattern=pattern) == \
                    np.array([index,index+half_len])).all(),
                msg=format_error_message(func)
            )

    @string_match_decorator(oned_string_matching_algorithms)
    def test_long_stream(self, func):
        np.random.seed(67+2)
        text = ''.join(np.random.choice(list('AGCT'), size=100000))
        pattern = ''.join(np.random.choice(list('AGTC'), size=3))

        self.assertTrue((func(text=text, pattern=pattern) == \
             np.array(boyermoore.boyer_moore_match_index(text=text, \
             pattern=pattern))).all(),
            msg=format_error_message(func))

    def test_chunk_sizes(self):
        text = "AAACCCAAA"
        chunk_size = 'm'
        pattern = "CC"
        self.assertTrue((
            fftmatch.fft_match_index_n_log_m(text, pattern, chunk_size) == \
            np.array(boyermoore.boyer_moore_match_index(text,pattern))).all())


class MultiGenomeTestRig(unittest.TestCase):
    @string_match_decorator(twod_string_matching_algorithms)
    def test_multi_genome_search(self, func):
        texts = ["ABCDABCDABCDABCD"]*4
        pattern = "ABCD"

        expected_output = np.array([[0,4,8,12]]*4)

        self.assertTrue((func(texts, pattern) == expected_output).all(),
                msg=format_error_message(func))

    @string_match_decorator(twod_string_matching_algorithms)
    def test_multi_genome_search_different_patterns(self, func):
        texts = ["ABCDABCDABCDABCD"]*4
        patterns = "ABCD", "BCD", "CD", "DAB"

        expected_outputs = [np.array([[0,4,8,12]]*4),
                           np.array([[1,5,9,13]]*4),
                           np.array([[2,6,10,14]]*4),
                           np.array([[3,7,11]]*4)]

        for i in range(len(patterns)):
            self.assertTrue((func(texts, patterns[i]) == \
                                expected_outputs[i]).all(),
                msg=format_error_message(func))

    @string_match_decorator(twod_string_matching_algorithms)
    def test_different_length_input_strings(self, func):
        #self.assertTrue(False)
        texts = ["ABCD", "ABC", "ABCDD"]
        pattern = "AB"

        expected_output = np.array([[0]]*3)

        self.assertTrue((func(texts, pattern) == expected_output).all())

        texts = ["ABCD", "ABC", "ABCDD"]
        pattern = "DD"

        expected_output = np.array([[], [], [3]])
        _pass = ndarrays_equal(func(texts, pattern), expected_output)

        self.assertTrue(_pass)

        texts = ["ABCD", "ABC", "ABCDD"]
        pattern = "DA"

        expected_output = np.array([[], [], []])
        _pass = ndarrays_equal(func(texts, pattern), expected_output)

        self.assertTrue(_pass)

    @string_match_decorator(twod_string_matching_algorithms)
    def test_long_stream(self, func):
        #if func == fftmatch.fft_match_index_n_sq_log_n_naive:
        #    return #You shall not pass
        np.random.seed(67+2)
        texts = np.random.choice(list('AGCT'), size=(10000,31)).tolist()
        texts = [''.join(_list) for _list in texts]
        pattern = ''.join(np.random.choice(list('AGTC'), size=3))

        expected_output = [boyermoore.boyer_moore_match_index(text=text,\
                            pattern=pattern) for text in texts]
        expected_output = np.array(expected_output)

        out = func(texts=texts, pattern=pattern)

        self.assertTrue(ndarrays_equal(out, expected_output),
                        msg=format_error_message(func))

    @string_match_decorator(twod_string_matching_algorithms)
    def test_example(self, func):
        s = "ACGACACAACACACACATAG"
        expected_output = np.array([[0]])
        out = func(texts=[s], pattern="ACG")
        print(expected_output)
        print(out)
        self.assertTrue(ndarrays_equal(out, expected_output),
                        msg = format_error_message(func))

if __name__ == '__main__':
    unittest.main()
