'''
Python is ideal for text classification, because of it's strong string class with powerful methods. Furthermore the regular expression module re of Python provides the user with tools, which are way beyond other programming languages.

The only downside might be that this Python implementation is not tuned for efficiency
'''
'''
Document representation counts how many word are there in your 
text, e.g ctrl+f chekcs how many specific word are there '''

import re
import os
import Modules.TextClassification

d1 = dict(a=4, b=5, d=8)
d2 = dict(a=1, d=10, e=9)

z= {k: d1.get(k, 0) + d2.get(k, 0) for k in (set(d1) | set(d2))}
print(z)
