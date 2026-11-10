# Writes the bytes of INPUT into OUTPUT as a C initializer list ("0x1c,0x00,..."), as
# `xxd -i < INPUT > OUTPUT` does, without needing xxd:
#
#     cmake -DINPUT=model.tflite -DOUTPUT=model.inc -P embed.cmake
file(READ "${INPUT}" bytes HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${bytes}")
file(WRITE "${OUTPUT}" "${bytes}\n")
