from version5_help.word_guidance import words_to_use_in_outgoing_messages_or_to_collect_advice_or_take_action

# Test with a context about improving effectiveness
result = words_to_use_in_outgoing_messages_or_to_collect_advice_or_take_action(
    "improving the program's effectiveness requires careful optimization",
    2,
    "user"
)
print("Result:", result)