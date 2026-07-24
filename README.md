A program that provides guidance.

I usually use it as follows:

from word_guidance import *
words_to_use_in_outgoing_messages_or_to_collect_advice_or_take_action(LONG_CONTEXT_OF_THE_SITUATION_AND_WHAT_SORT_OF_GUIDANCE_IM_LOOKING_FOR,
2, LIST_OF_RELEVANT_PEOPLE_IN_THIS_SITUATION)

It spits out a list of words that I can then use in various ways.

For example, for the redesign of all TalkDoc websites, I asked for words that would populate the mood board for the design of the new TalkDoc webpages.
It did that really well, as can be evidenced by the current TalkDoc.com website, which I updated over the past 24 hours using this technique.

## Version 5 (improved)

The same calls work as before, but the words are now drawn from curated,
speakable banks (guidance / advice / action) instead of the raw 235,000-word
system dictionary, every word carries a one-line gloss, and results are
deterministic for a given question (pass vary=True for variety). Also new:

from word_guidance import guidance_report
report = guidance_report(CONTEXT, 2, RELEVANT_PEOPLE)

The report contains: words_to_use_in_outgoing_messages, advice_to_collect,
actions_to_take, a gita_resonance (a word from the Bhagavad Gita with its
verse — the corpus the old version loaded but never used), definitions for
every word, a one-paragraph reading, a deterministic signature for auditing,
and a valid flag confirming every word passed validation.

From the command line:

python3 word_guidance.py "your situation and what guidance you want" 2 "the people involved"
python3 word_guidance.py --check   # sanity self-check (see meta/GOALS.md)
