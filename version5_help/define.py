
#!/usr/bin/env python

"""
# Version
2021-08-31 
# Tested on 
Python 3.9
# Originally Modified from
http://macscripter.net/viewtopic.php?id=26675
http://apple.stackexchange.com/questions/90040/look-up-a-word-in-dictionary-app-in-terminal
# HowTo
chmod 755 define.py # make file executable
alias define="'/Users/me/Script/define.py'" # => add to .bash_profile
-> then (from command line)
define recursive # or any other word
-> output ..
Dictionary entry for:
recursive | rɪˈkəːsɪv | adjective characterized by recurrence or repetition.
 • Mathematics & Linguistics relating to or involving the repeated application of a rule, definition, or procedure to successive results: this restriction ensures that the grammar is recursive.
 • Computing relating to or involving a program or routine of which a part requires the application of the whole, so that its explicit interpretation requires in general many successive executions: a recursive subroutine.
DERIVATIVES
 recursively | rɪˈkəːsɪvli | adverb
ORIGIN
 late 18th century (in the general sense): from late Latin recurs- ‘returned’ (from the verb recurrere ‘run back’) + -ive. Specific uses have arisen in the 20th century.
---------------------------------
"""



import sys

try:
	from DictionaryServices import *
except:
	print("WARNING: The pyobjc Python library was not found. You can install it by typing: 'pip install -U pyobjc'")
	print("..................\n")
	

try:
	from colorama import Fore, Back, Style
except:
	print("WARNING: The colorama Python library was not found. You can install it by typing: 'pip install colorama'")
	print("..................\n")




	
def main():
	"""
	define.py
	
	Access the default OSX dictionary
	
	2021-08-31
	-improved so to work with Python 3.9
	2015-11-27
	-added colors via colorama
	
	"""
	try:
		searchword = sys.argv[1]
	except IndexError:
		errmsg = 'You did not enter any terms to look up in the Dictionary.'
		print(errmsg)
		sys.exit()
	wordrange = (0, len(searchword))
	dictresult = DCSCopyTextDefinition(None, searchword, wordrange)
	if not dictresult:
		errmsg = "'%s' not found in Dictionary." % (searchword)
		print(errmsg)
	else:
		# print(dictresult)
		s = dictresult	

		#split numbered items up to ten (hackish..)
		for n in reversed(range(1, 11)):
			new_n = doColor(f"\n{str(n)} ", "red")			
			s = s.replace(f" {str(n)} ", new_n) 

		for x in ["noun", "adverb", "adjective"]:
			particle = doColor(x, "green")	
			s = s.replace(x, particle)	

		bullet = doColor("\n •", "red")	
		s = s.replace('•', bullet)	# bullet
		
		phrases_header = doColor("\n\nPHRASES\n", "important")
		s = s.replace('PHRASES', phrases_header)
		
		phrasal_verbs_header = doColor("\n\nPHRASAL VERBS\n", "important")
		s = s.replace('PHRASAL VERBS', phrasal_verbs_header)

		derivatives_header = doColor("\n\nDERIVATIVES\n", "important")
		s = s.replace('DERIVATIVES', derivatives_header)
		
		origin_header = doColor("\n\nORIGIN\n", "important")
		s = s.replace('ORIGIN', origin_header)
		
		print(doColor("Dictionary entry for:\n", "red"))
		print(s)
		print("\n---------------------------------")




def doColor(s, style=None):
	"""
	util for returning a colored string
	if colorama is not installed, FAIL SILENTLY
	"""
	try:
		if style == "comment":
			s = Style.DIM + s + Style.RESET_ALL
		elif style == "important":
			s = Style.BRIGHT + s + Style.RESET_ALL
		elif style == "normal":
			s = Style.RESET_ALL + s + Style.RESET_ALL	
		elif style == "red":
			s = Fore.RED + s + Style.RESET_ALL	
		elif style == "green":
			s = Fore.GREEN + s + Style.RESET_ALL			
	except: 
		pass
	return s



if __name__ == '__main__':
	main()
