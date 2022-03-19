import numpy as np
from collections import OrderedDict, Counter

class Lab1(object):
    def parse_reads_illumina(self,reads) :
        '''
        Input - Illumina reads file as a string
        Output - list of DNA reads
        '''
        #start code here
        dna_reads = []
        
        lines = reads.splitlines()
        
        for i in range(1, len(lines), 4):
            dna_reads.append(lines[i])
        
        return dna_reads
        #end code here

    def unique_lengths(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - set of counts of reads
        '''
        #start code here
        counts = set()
        
        for dna_read in dna_reads:
            counts.add(len(dna_read))
        
        return counts
        #end code here

    def check_impurity(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - list of reads which have impurities, a set of impure chars 
        '''
        #start code here
        impure_reads = []
        impure_chars = set()

        for dna_read in dna_reads:
            impure = False
            
            for char in dna_read:
                if char.lower() not in ['a', 'c', 'g', 't']:
                    impure_chars.add(char)
                    impure = True
            
            if impure == True:
                impure_reads.append(dna_read)
        
        return impure_reads, impure_chars
        #end code here

    def get_read_counts(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - dictionary with key as read and value as the no. of times it occurs
        '''
        #start code here
        return Counter(dna_reads)
        #end code here

    def parse_reads_pac(self,reads_pac) :
        '''
        Input - pac bio reads file as a string
        Output - list of dna reads
        '''
        #start code here
        dna_reads_pac = []
        
        for line in reads_pac.split('>')[1:]:
            dna_reads_pac.append(''.join(line.split('\n')[1:]))
        
        return dna_reads_pac
        #end code here