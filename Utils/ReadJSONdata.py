# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:31:13 2017

@author: DominiqueFournier
"""

# Read JSON file
import json

fileDir = "C:\\Users\\DominiqueFournier\\Documents\\giftoolsii-issues"

with open(fileDir + '\\db-1.0.json') as json_data:
    d = json.load(json_data)

issues = d['issues']
comments = d['comments']

# for ii, issue in enumerate(issues):

#     content = issue['content']
#     if content.find('Nick') != -1:
#         print('Found Nick in issue: ' + "#" + str(issue['id']) + ": " + issue['title'])


# for comment in comments:
#    if comment["issue"] == 5:
#        print(comment['content'])
