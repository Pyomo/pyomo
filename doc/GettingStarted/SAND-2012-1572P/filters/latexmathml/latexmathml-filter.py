.Code Filter out latex math environments, and replace with <blockquote>
[code,python]
----------------------------------------------
’’’ A multi-line comment.’’’
def sub_word(mo):
    ’’’ Single line comment.’’’
    word = mo.group(’word’) # Inline comment
    print "HERE", word
    if word == '\\[':
        return '<Xblockquote>'
    elif word == '\\]':
        return '</Xblockquote>'
    else:
        return word
----------------------------------------------
