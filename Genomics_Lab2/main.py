import numpy as np
from collections import OrderedDict
import re

class Lab2(object):
    
    def smith_waterman_alignment(self,s1,s2,penalties) :
        '''
        Input - two sequences and a dictionary with penalities for match, mismatch and gap
        Output - an integer value which is the maximum smith waterman alignment score
        '''
        #start code here
        match = penalties['match']
        mismatch = penalties['mismatch']
        gap = penalties['gap']
    
        score_matrix, start_pos = create_score_matrix(len(s1) + 1, len(s2) + 1, s1, s2, penalties)
        s1_aligned, s2_aligned = traceback(score_matrix, start_pos, s1, s2)
        
        max_score = 0
        for a in s1_aligned.lower():
            for b in s2_aligned.lower():
                if a == b:
                    max_score += match
                elif a == '-' or b == '-':
                    max_score += gap
                else:
                    max_score += mismatch
        
        return max_score
        #end code here

    def print_smith_waterman_alignment(self,s1,s2,penalties) :
        '''
        Input - two sequences and a dictionary with penalities for match, mismatch and gap
        Output - a tuple with two strings showing the two sequences with '-' representing the gaps
        '''
        #start code here
        score_matrix, start_pos = create_score_matrix(len(s1) + 1, len(s2) + 1, s1, s2, penalties)
        s1_aligned, s2_aligned = traceback(score_matrix, start_pos, s1, s2)
        
        return (s1_aligned, s2_aligned)
        #end code here

    def find_exact_matches(self,list_of_reads,genome):
        
        '''
        Input - list of reads of the same length and a genome fasta file (converted into a single string)
        Output - a list with the same length as list_of_reads, where the ith element is a list of all locations (starting positions) in the genome where the ith read appears. The starting positions should be specified using the "chr2:120000" format
        '''
        
        #start code here
        d = []
        result = []
        
        file = genome.splitlines()
        header = []
        for line in file:
            if '>' in line:
                file.remove(line)
                header.append(line)
        
        for pattern in list_of_reads:
            for i in range(len(file)):
                result.append(str(header[i]:re.search(pattern.lower(), file[i].lower())))
        
        return result
        #end code here
       
    
    def find_approximate_matches(self,list_of_reads,genome):
        '''
        Input - list of reads of the same length and a genome fasta file (converted into a single string)
        Output -  a list with the same length as list_of_reads, where the ith element is a list of all locations (starting positions) in the genome which have the highest smith waterman alignment score with ith read in list_of_reads
        '''
        
        #start code here
        return None
        #end code here
        
# end of Lab 2 class
# Helper Functions
def create_score_matrix(rows, cols, seq1, seq2, penalties):
    '''
    Create a matrix of scores representing trial alignments of the two sequences
    '''
    score_matrix = [[0 for col in range(cols)] for row in range(rows)]
    
    # Fill the scoring matrix.
    max_score = 0
    max_pos   = None    # The row and columbn of the highest score in matrix.
    for i in range(1, rows):
        for j in range(1, cols):
            score = calc_score(score_matrix, i, j, seq1, seq2, penalties)
            if score > max_score:
                max_score = score
                max_pos   = (i, j)
    
            score_matrix[i][j] = score
    
    assert max_pos is not None, 'the x, y position with the highest score was not found'
    
    return score_matrix, max_pos
    
def calc_score(matrix, x, y, seq1, seq2, penalties):
    '''
    Helper function to calculate score for a given x, y position in the scoring matrix
    '''
    match = penalties['match']
    mismatch = penalties['mismatch']
    gap = penalties['gap']
    
    similarity = match if seq1[x - 1] == seq2[y - 1] else mismatch

    diag_score = matrix[x - 1][y - 1] + similarity
    up_score   = matrix[x - 1][y] + gap
    left_score = matrix[x][y - 1] + gap

    return max(0, diag_score, up_score, left_score)

def traceback(score_matrix, start_pos, seq1, seq2):
    '''
    Find the optimal path through the matrix
    '''

    END, DIAG, UP, LEFT = range(4)
    aligned_seq1 = []
    aligned_seq2 = []
    x, y = start_pos
    move = next_move(score_matrix, x, y)
    
    while move != END:
        if move == DIAG:
            aligned_seq1.append(seq1[x - 1])
            aligned_seq2.append(seq2[y - 1])
            x -= 1
            y -= 1
        elif move == UP:
            aligned_seq1.append(seq1[x - 1])
            aligned_seq2.append('-')
            x -= 1
        else:
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[y - 1])
            y -= 1

        move = next_move(score_matrix, x, y)

    aligned_seq1.append(seq1[x - 1])
    aligned_seq2.append(seq1[y - 1])

    return ''.join(reversed(aligned_seq1)), ''.join(reversed(aligned_seq2))


def next_move(score_matrix, x, y):
    ''' 
    Helper function to determine next move for traceback
    '''
    diag = score_matrix[x - 1][y - 1]
    up   = score_matrix[x - 1][y]
    left = score_matrix[x][y - 1]
    if diag >= up and diag >= left:     # Tie goes to the DIAG move.
        return 1 if diag != 0 else 0    # 1 signals a DIAG move. 0 signals the end.
    elif up > diag and up >= left:      # Tie goes to UP move.
        return 2 if up != 0 else 0      # UP move or end.
    elif left > diag and left > up:
        return 3 if left != 0 else 0    # LEFT move or end.
    else:
        # Execution should not reach here.
        raise ValueError('invalid move during traceback')