import png
import sys


# given data which is a list of lists of 0 and one write that to a png file with file_name
# example input would be [[0, 1, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0, 1]]
def write_png(data, file_name):
    writer = png.Writer(len(data[0]), len(data), greyscale=True, bitdepth=1)
    file = open(file_name, "wb")
    writer.write(file, data)
    file.close()


# add 0 in front of the string until it has 8 digits
# so 111 is converted into 0000111
def ensure_eight_bits(binary_letter):
    needed_zeros = 8 - len(binary_letter)
    for i in range(needed_zeros):
        binary_letter = "0" + binary_letter
    return binary_letter


# given a string of 0 and 1 like 0000111 convert that to a list of integer 0 and 1
# so 0000111 becomes [0,0,0,0,1,1,1]
def binary_string_to_int_list(binary_string):
    result = []
    for binary_char in binary_string:
        result.append(int(binary_char))
    return result


# given a secret message convert it to a list of integer lists
# like Hi = [[0, 1, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0, 1]]
def string_to_binary_array(string_message):
    result = []
    for letter in string_message:
        # ord converts a letter to an ascii integer see https://www.asciitable.com/
        ascii_int_letter = ord(letter)
        # now convert the ascii integer into a binary string
        binary_string_letter = "{0:b}".format(ascii_int_letter)
        # add 0 to the front so that we always have 8 digits 0 or 1
        binary_string_letter = ensure_eight_bits(binary_string_letter)
        # convert the string to a list of integers
        # so H = [0, 1, 0, 0, 1, 0, 0, 0]
        binary_int_list = binary_string_to_int_list(binary_string_letter)
        result.append(binary_int_list)
    return result


# the program starts here, above this is functions

# check to make sure they provided a secret message. Tell them how to call this if they didn't
if len(sys.argv) != 2:
    print('Usage python barcodeWriter.py "the secret message"')
    exit()

# get the message from the command line argument
message = sys.argv[1]

# convert the message to binary example result would be [[0, 1, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0, 1]]
binaryDigits = string_to_binary_array(message)

# write the binary to a png file
# example input would be [[0, 1, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0, 1]]
write_png(binaryDigits, "code.png")
