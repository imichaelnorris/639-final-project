import fftmatch as fft
import boyermoore as bm
import argparse
import collections

parser = argparse.ArgumentParser(description='Search for a substring in a \
genome')

# Algorithm flag: Options= nlogn, nlogm, boyer moore; Default=nlogn
parser.add_argument('-a','--algorithm', choices=["nlogn", "nlogm", "boyermoore"],
                    default='nlogn', nargs='?', help='The algorithm that you \
want to run the search on. Default=nlogn')

# Pattern arg: substring to search genomes for.
parser.add_argument('pattern', help='The pattern that you want to search for in\
 the genome(s)', nargs=1)

# Genome arg: Genomes to search
parser.add_argument('genomes', nargs='+',
                    help='1 or more fastq files (.fa). \
You can also pass in 1 or more genomes, separated by spaces')

args = parser.parse_args()
genomes = {}

# Scan files and store the title and genome string in genomes dictionary
for genome_fn in args.genomes:
    with open(genome_fn) as gn:
        title = gn.readline().rstrip()
        genome = ''
        for line in gn:
            genome += line.rstrip()
    genomes[title] = genome

sorted_genomes = collections.OrderedDict(sorted(genomes.items(),
                                      key=lambda t: t[0]))
genome_strings = sorted_genomes.values()
genome_titles = sorted_genomes.keys()

# Parse args
if args.algorithm == 'nlogn':
    matches = fft.fft_match_index_n_sq_log_n(genome_strings, args.pattern[0])
    for i,match in enumerate(matches):
        print genome_titles[i], ': Found matches at indices', match
    pass
elif args.algorithm == 'nlogm':
    matches = fft.fft_match_index_n_sq_log_m(genome_strings, args.pattern[0])
    for i,match in enumerate(matches):
        print genome_titles[i], ': Found matches at indices', match
    pass
elif args.algorithm == 'boyermoore':
    matches = bm.boyer_moore_mult_match_index(genome_strings, args.pattern[0])
    for i,match in enumerate(matches):
        print genome_titles[i], ': Found matches at indices', match
    pass
