import png
import sys


def write_png(data, file_name):
    writer = png.Writer(len(data[0]), len(data), greyscale=True, bitdepth=1)
    file = open(file_name, "wb")
    writer.write(file, data)
    file.close()


def ensure_eight_bits(binary_letter):
    needed_zeros = 8 - len(binary_letter)
    for i in range(needed_zeros):
        binary_letter = "0" + binary_letter
    return binary_letter


def binary_string_to_int_list(binary_string):
    result = []
    for binary_char in binary_string:
        result.append(int(binary_char))
    return result


def string_to_binary_array(string_message):
    result = []
    for letter in string_message:
        ascii_letter = ord(letter)
        binary_string_letter = "{0:b}".format(ascii_letter)
        binary_string_letter = ensure_eight_bits(binary_string_letter)
        binary_int_list = binary_string_to_int_list(binary_string_letter)
        result.append(binary_int_list)
    return result


if len(sys.argv) != 2:
    print('Usage python barcodeWriter.py "the secret message"')
    exit()
message = sys.argv[1]
binary = string_to_binary_array(message)

write_png(binary, "code.png")
