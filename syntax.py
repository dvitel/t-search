'''
Syntactic indexes - tries to avoid recreation of same terms
Also, contains linear form of terms
'''

#TODO: term stats - something like tries on terms - or n-grams on sequences 
#NOTE: with stats on trees and their position closer to solution we can conclude betternes of some syntax constructs
#Therefore, decision what to add to a tree could be based on stats


from term import Term

class TermIndex:

    def __init__(self):
        self.terms = []


# class TermTrie:
#     ''' stores terms by common prefixes '''    

#     def __init__(self, operator):
#         self.operator: str = None #
#         self.roots: list['TermTrie'] = []
#         self.op_map = {} # maps operator to corresponding root

#     def add(self, term: Term):