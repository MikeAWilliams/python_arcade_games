import png
from bitstring import BitArray

def binary_list_to_int_list(binary_list):
    result = []
    for row in binary_list:
        int_row = []
        for binary_digit in row:
            int_row.append(int(binary_digit))
        result.append(int_row)
    return result


def read_png(file_name):
    with open(file_name, "rb") as file:
        reader = png.Reader(file=file)
        tup = reader.read()
        binary_list = list(tup[2])
        return binary_list_to_int_list(binary_list)


png_int_list = read_png("code.png")
print(png_int_list)

result = ''
for letter_list in png_int_list:
    letter_binary_string = ''
    for binary_digit in letter_list:
        letter_binary_string = letter_binary_string + str(binary_digit)
    print(letter_binary_string)
    int_for_char = BitArray(bin=letter_binary_string).int
    result = result + chr(int_for_char)
print(result)