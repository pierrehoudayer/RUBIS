#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:17:53 2022

@author: phoudayer
"""

class DotDict(dict):  
    """dot.notation access to dictionary attributes"""      
    def __getattr__(*args):        
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val     
    __setattr__ = dict.__setitem__     
    __delattr__ = dict.__delitem__ 