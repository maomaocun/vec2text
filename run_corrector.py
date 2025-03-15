import vec2text
# corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")
inversion_model = vec2text.models.InversionModel.from_pretrained("jxm/gtr__nq__32")
corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained("jxm/gtr__nq__32__correct")

corrector = vec2text.load_corrector(inversion_model, corrector_model)
# input_text = ['It was the age of incredulity, the age of wisdom, the age of apocalypse, the age of apocalypse, it was the age of faith, the age of best faith, it was the age of foolishness']
input_text = ['I am a second-year PhD student, pursuing my doctorate at Cornell Tech in New York City']

result_in_step = vec2text.invert_strings(
    input_text,
    corrector=corrector,
)

result_in_steps = vec2text.invert_strings(
    input_text,
    corrector=corrector,
    num_steps=20,
)

result_in_steps_with_sequence_beam_width = vec2text.invert_strings(
    input_text,
    corrector=corrector,
    num_steps=20,
    sequence_beam_width=4,
)
print("Invert strings:")
print(input_text)

print("Result in step:")
print(result_in_step)

print("\nResult in steps:")
print(result_in_steps)

print("\nResult in steps with sequence beam width:")
print(result_in_steps_with_sequence_beam_width)