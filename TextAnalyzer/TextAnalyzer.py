import sys
import argparse
import numpy as np
from pyspark import SparkContext


def toLowerCase(s):
    """ Convert a sting to lowercase. E.g., 'BaNaNa' becomes 'banana'
    """
    return s.lower()

def stripNonAlpha(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """
    return ''.join([c for c in s if c.isalpha()])

def seqOp(topK, x, k):
    '''Perform a sequence operation between the input and output types
       locally in the partition.
         - topK is the output type
         - x is the input type
         - k is the length of the output list'''
    topK.append(x)
    topK.sort(key=lambda (w, n): n, reverse=True)
    if len(topK) > k:
        topK.pop()
    return topK

def merge(topK1, topK2, k):
    '''Perform a merge between partition output types (list).
         - topK1 is one output type
         - topK2 is a second output type
         - k is the output (list) length'''
    merged = topK1+topK2
    merged.sort(key=lambda (w, n): n, reverse=True)
    if len(merged) > k:
        merged = merged[:k]
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text Analysis through TFIDF computation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('mode', help='Mode of operation',choices=['TF','IDF','TFIDF','SIM','TOP']) 
    parser.add_argument('input', type=str, help='Input file or list of files.')
    parser.add_argument('output', help='File in which output is stored')
    parser.add_argument('--master',default="local[20]",help="Spark Master")
    parser.add_argument('--idfvalues',type=str,default="idf", help='File/directory containing IDF values. Used in TFIDF mode to compute TFIDF')
    parser.add_argument('--other',type=str,help = 'Score to which input score is to be compared. Used in SIM mode')
    args = parser.parse_args()
  
    sc = SparkContext(args.master, 'Text Analysis')


    if args.mode=='TF':
        # Read text file at args.input, compute TF of each term, 
        # and store result in file args.output. All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings, i.e., "" 
        # are also removed

        # 1) Read in the input file(s). This creates a list of strings.
        #    Each string is a line of the input file.
        # 2) Split the strings at every white space, resulting in a 2D list.
        #    Apply a flatMap in order to have a 1D list containing all words
        #    of input file. 
        # 3) Convert all words to lower case.
        # 4) Map the stripNonAlpha function to all words in the list.
        # 5) Format the list of words into a list of tuple containig the word
        #    and an integer representing its frequency.
        # 6) Filter out the tuples containing an empty string word.
        # 7) Reduce the tuples by key; add all the frequencies of the same word.
        # 8) Save the list of tuples into a text file. 
        f = sc.textFile(args.input) \
              .flatMap(lambda w: w.split()) \
              .map(lambda w: w.lower()) \
              .map(stripNonAlpha) \
              .map(lambda w: (w, 1)) \
              .filter(lambda (w,i): w != '') \
              .reduceByKey(lambda x, y: x + y) \
              .saveAsTextFile(args.output)


    if args.mode=='TOP':
        # Read file at args.input, comprizing strings representing pairs of the 
        # form (TERM,VAL), where TERM is a string and VAL is a numeric value. 
        # Find the pairs with the top 20 values, and store result in args.output

        # 1) Read in file from input.
        # 2) Evaluate the list of strings into their tuple expressions.
        # 3) Sort the list of tuples in descending order w.r.t the word
        #    frequencies.
        # 4) Aggregate the top 20 words with highest frequency. This aggregate 
        #    method uses two seperate functions: seqOp and merge. seqOp takes 
        #    in the input type -an element- and the output type -the list. merge 
        #    takes in two output types: two lists. The seqOp is performed locally 
        #    at each partition, and the merge is performed at the communication 
        #    stage between partitions.
        # 5) We open an output file and write to it the top 20 words with highest 
        #    frequency. 
        f = sc.textFile(args.input) \
              .map(lambda t: eval(str(t))) \
              .sortBy(lambda (word, i): i, ascending=False) \
              .aggregate([], lambda li, el: seqOp(li, el, 20), lambda li1, li2: merge(li1, li2, 20))

        with open(args.output, 'w') as fout:
            for t in f:
                fout.write(str(t)+'\n')
        

    if args.mode=='IDF':
        # Read list of files from args.input, compute IDF of each term,
        # and store result in file args.output.  All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings ""
        # are removed

        
        # 1) Read in whole text files from input
        # 2) Keep the total count of documents
        # 3) Apply a map to the list of tuples containing the file name and the
        #    document text. Each document text is split by their whitespaces.
        #    For each word in the list of newly split words we stripNonAlpha
        #    characters and lower all potential uppercase characters. The list
        #    of words is converted into a set in order to remove any duplicate
        #    words, and then the set is converted back into a list. The result
        #    is a list of tuples containing the file name and a list of unique
        #    words from that document.
        # 4) The list of tuples containing file names and document content is
        #    converted into a list of tuples containing a word and a counter 
        #    for each document. Each word is instantiated with a counter of 1
        #    given its occurence in the given file. 
        # 5) The list of tuples per document is flattened into a single list
        #    of tuples.
        # 6) The tuples are redced by their key. All word counters are added
        #    for the same word.  
        # 7) Empty strings are filtered out.
        # 8) Compute the inverse document frequency of each word given the 
        #    total number of documents saved earlier and the word count found
        #    in the tuple.
        # 9) Save the list of tuples to output.
        corpus = sc.wholeTextFiles(args.input)
        ndocs = corpus.count()
        corpus.map(lambda (path, content): (str(path), 
                                            list(set([stripNonAlpha(c).lower() 
                                                      for c in content.split()])))) \
              .map(lambda (path, content): [(word, 1) for word in content]) \
              .flatMap(lambda t: t) \
              .reduceByKey(lambda x,y: x+y) \
              .filter(lambda (w,i): w != '') \
              .map(lambda (word, n): (word, np.log(1.0*ndocs/n))) \
              .saveAsTextFile(args.output)


    if args.mode=='TFIDF':
        # Read  TF scores from file args.input the IDF scores from file args.idfvalues,
        # compute TFIDF score, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase letter-only string and VAL is a numeric value.  
        
        # 1) Read in the term frequency file(s) from input
        # 2) Read in the inverse document frequency file(s) from idfvalues
        # 3) Join the TF and IDF rdds to obtain a new rdd containig the common
        #    key tuples from both rdds. The resulting joint rdd contains a list
        #    of tuples. The first element is the word, and the second is a tuple
        #    with the word's frequency and inverse document frequency.
        # 4) Map the list of tuples to contain a tuple with the word and the
        #    TFIDF value.
        # 5) Save the list of tuples to output.
        tf = sc.textFile(args.input) \
               .map(lambda t: eval(t))
        idf = sc.textFile(args.idfvalues) \
                .map(lambda t: eval(t))
        tf.join(idf) \
          .map(lambda (w, (freq, ifreq)): (w, freq*ifreq)) \
          .saveAsTextFile(args.output)
            
 
    if args.mode=='SIM':
        # Read  scores from file args.input the scores from file args.other,
        # compute the cosine similarity between them, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL), 
        # where TERM is a lowercase, letter-only string and VAL is a numeric value. 
       
        # 1) Read in the TFIDF file(s) from input.
        # 2) Read in the TFIDF file(s) from other.
        # 3) Join both TFIDF files and store the new rdd into num.
        # 4) Apply a map on num where the two TFIDF scores of a word are 
        #    multiplied with each other. Remove the word and keep the score.
        # 5) Reduce the all the scores by summing them up.
        # 6) Apply a map to each TFIDF from input. Each word's scores are
        #    squared. Forget the word itself.
        # 7) Reduce the scores from input by summing them up.
        # 8) Apply a map to each TFIDF from other. Each word's scores are
        #    squared. Forget the word itself.
        # 9) Reduce the scores from other by summing them up.
        # 10) Compute the final cosine similarity score from the previously
        #     calculated scores. 
        # 11) Save the similarity score to output. 
        tfidf1 = sc.textFile(args.input) \
                   .map(lambda t: eval(t))
        tfidf2 = sc.textFile(args.other) \
                   .map(lambda t: eval(t)) \
        
        num = tfidf1.join(tfidf2) \
                      .map(lambda (w, (n1, n2)): n1*n2) \
                      .reduce(lambda x,y: x+y)

        tfidf1 = tfidf1.map(lambda (w, n): n**2) \
                       .reduce(lambda x,y: x+y)
        tfidf2 = tfidf2.map(lambda (w, n): n**2) \
                       .reduce(lambda x,y: x+y)

        sim = num / ((tfidf1 * tfidf2)**0.5)

        with open(args.output, 'w') as f:
            f.write(str(sim))
