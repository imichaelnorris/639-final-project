'''
Implementation of the FFT match-index problem for finding one substring inside
a source text (genome).
'''
import numpy as np

def string_to_binary_array(s, size=None):
    #TODO: check dtype for the algorithm
    #A,C,G,T = [np.zeros(len(s) if not size else size) for _ in range(4)]
    t = np.zeros(len(s) if not size else size)
    for index, val in enumerate(s):
        t[index] = float(ord(val))

    return t

#TODO: replace this with one of our other faster match-index solving algs from
# lecture notes or homeworks
def naive_string_match_index(text, pattern):
    '''A naive string matching algorithm that solves the match-index problem.
    The match-index problem provides a list of indices where a pattern matches
    a text.
    arguments:
      text: the text that you are interested in find
      pattern: the pattern that may be contained in multiple locations inside
        the text
    returns: a list containing the 0-based indices of matches of pattern in text
    '''
    pattern_len = len(pattern)
    matches = []
    for i in range(len(text)-len(pattern)+1):
        if text[i:i+pattern_len] == pattern:
            matches.append(i)
    return matches

def fft_match_index_n_log_n(text, pattern):
    '''Does the n log n FFT pattern matching algorithm.  This solves the match
    index problem by returning a list of indices where the pattern matches the
    text.

    Does cross-correlation solving the following equation:
    S_{i} = \Sigma_{j=1}^{m} (p_{j}^{3} t_{i+j-1} - 2p_{j}^{2}t_{i+j-1}^{2}
                              + p_{j}t_{i+j-1}^{3})
    This can be solved in Fourier space using FFT's of each of the three terms.

    S_{i} = 0 when there is a match between the pattern and text at that
    location.

    TODO: cite papers

    arguments:
      text: the text that you are interested in find
      pattern: the pattern that may be contained in multiple locations inside
        the text
    returns: a list containing the 0-based indices of matches of pattern in text
    '''

    #Note: len(rfft(something)) != len(something) for general case

    n = len(text)
    m = len(pattern)
    binary_encoded_text = string_to_binary_array(text)

    #TODO: for binary_encoded_text and pattern, if the char is equal to the
    # don't care character, then set the float value to 0.0
    text = binary_encoded_text
    textSq = text * text
    textCube = textSq * text

    binary_encoded_pattern = string_to_binary_array(pattern,size=len(text))

    assert len(binary_encoded_text) == len(binary_encoded_pattern)

    pattern = binary_encoded_pattern
    patternSq = pattern * pattern
    patternCube = patternSq * pattern

    textKey = np.fft.rfft(text)
    textSqKey = np.fft.rfft(textSq)
    textCubeKey = np.fft.rfft(textCube)

    patternKey = np.fft.rfft(pattern)
    patternSqKey = np.fft.rfft(patternSq)
    patternCubeKey = np.fft.rfft(patternCube)

    #there are three terms.  Since fft(key) is Linear, we will IFT each
    #individually
    outTerm1Key = patternCubeKey * textKey
    outTerm2Key =  patternSqKey * textSqKey
    outTerm3Key = patternKey * textCubeKey

    outTerm1 = np.fft.irfft(outTerm1Key)
    outTerm2 = -2*np.fft.irfft(outTerm2Key)
    outTerm3 = np.fft.irfft(outTerm3Key)

    #TODO: may need to rotate this
    out = outTerm1 + outTerm2 + outTerm3

    #this should be 0 if match
    #TODO: figure out the difference between exact and inexact.
    #I think true matches where 0 and possible matches below this threshold
    return np.ndarray.tolist(np.where(abs(out) < 1.0e-6))

if __name__ == '__main__':
    #f = open('1d.txt')
    #text = f.read().replace('\n', '')
    #pattern = 'ACG'
    #text = "ABCDABCDABCDABCD"
    #pattern = "ABCD"
    text = "ABCD"
    pattern = "D"

    out = fft_match_index_n_log_n(text, pattern)
    #print out, naive_string_match_index(text, pattern)
    #assert out == naive_string_match_index(text, pattern)
    print(out)