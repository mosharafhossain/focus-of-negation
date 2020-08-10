#!/usr/bin/env python

import sys
from optparse import OptionParser

def read_focus(file_name):
    foci = []
    f = open(file_name)
    focus = []
    for l in f.readlines():
        l = l.strip()
        if len(l) == 0:
            foci.append(focus)
            focus = []
        else:
            focus.append(l.split()[-1])
    return foci

def calculate_performance(output, gold, num_sent, f_verbose):
    assert len(output) == len(gold), \
           "Different number of tokens for sentence %s: %s %s" % (num_sent, len(output), len(gold))
    
    o_focus = 0
    g_focus = 0
    correct = 1
    for o, g in zip(output, gold):
        if o == "FOCUS":
            o_focus = 1
        if g == "FOCUS":
            g_focus = 1
            
        if o != g:
            correct = 0
            if f_verbose:
                f_verbose.write("%s\t%s\t%s\t%s\n" % (num_sent, o, g, "wrong"))
        elif f_verbose:
            f_verbose.write("%s\t%s\t%s\t%s\n" % (num_sent, o, g, "ok"))
                
    if f_verbose:
        f_verbose.write("\n")

    if o_focus == 0 or g_focus == 0:
        correct = 0

    return o_focus, g_focus, correct    

def divide(top, bottom):
    if bottom == 0:
        return 0
    return float(top)/bottom

def f_score(p, r):
    return divide(2 * p * r, p + r)

def main(output, gold, f_verbose):
    output_focus = read_focus(output)
    gold_focus   = read_focus(gold)
    assert len(output_focus) == len(gold_focus), \
           "Number of sentences don't match: %s %s" % (len(output_focus), len(gold_focus))

    results = [0, 0, 0]
    for o, g, i in zip(output_focus, gold_focus, range(len(gold_focus))):
        results = [sum(pair) for pair in zip(results,             
                                             calculate_performance(o, g, i, f_verbose))]

    p = divide(results[2], results[0]) * 100
    r = divide(results[2], results[1]) * 100
    f = f_score(p, r)
    
    print ("Precision  %.2f [%s/%s]" % (p, results[2], results[0]))
    print ("Recall     %.2f [%s/%s]" % (r, results[2], results[1]))
    print ("F-score    %.2f" % f)

if __name__ == "__main__":
    usage = "usage: %prog [options] SYSTEM_OUTPUT GOLD"
    parser = OptionParser(usage=usage)

    parser.add_option("-v", "--verbose_file",
                      help="Create a file spelling out errors.")
    
    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("Incorrect number of arguments")
    
    f_verbose = None
    if options.verbose_file:
        f_verbose = open(options.verbose_file, 'w')
        f_verbose.write("sent\toutput\tgold\n")        
    
    main(args[0], args[1], f_verbose)
    
    if options.verbose_file:
        f_verbose.close()

    
