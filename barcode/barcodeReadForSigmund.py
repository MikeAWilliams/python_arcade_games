import png


# convert a list of binary lists to list of integer lists
def binary_list_to_int_list(binary_list):
    result = []
    for row in binary_list:
        int_row = []
        for binary_digit in row:
            int_row.append(int(binary_digit))
        result.append(int_row)
    return result


# open the png file, read its conents as a list of binary lists
# then convert that into integers
# result will be [[0, 1, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1, 1, 1],...]
# each innter [] will have 8 binary digits (0 or 1)
def read_png(file_name):
    with open(file_name, "rb") as file:
        reader = png.Reader(file=file)
        tup = reader.read()
        binary_list = list(tup[2])
        return binary_list_to_int_list(binary_list)


# convert the lists of 0 and 1 to strings of 0 and 1
# the data will be the same but it will be easier to read
def int_list_to_string_list(int_list):
    result = []
    for letter_list in int_list:
        letter_binary_string = ""
        for binary_digit in letter_list:
            letter_binary_string = letter_binary_string + str(binary_digit)
        result.append(letter_binary_string)
    return result


# this setp is meant to convert the binary string into a number on the ascii chart
# https://www.asciitable.com/
# this doesn't work because I didn't write binary_string_to_int
# you need to fix binary_string_to_int for this to work
def string_list_to_ascii(string_list):
    result = []
    for letter_as_binary_string in string_list:
        result.append(binary_string_to_int(letter_as_binary_string))
    return result


# The code above this works fine. you can just ignore it if you like.
# However, the code below needs to be filled in to get a result


# this function needs to use binary math to convert the binary string into an integer
# because each character is 8 bits long the biggest number it can be is 255
# an example of the math for this is https://www.shiksha.com/online-courses/articles/how-to-convert-binary-to-decimal/
# a quick note about how it works. Each binary 0 or 1 is a power of two
# the sum of those is the result, but it only sums if the digit is a 1
# so 11111111 is
# 2**7 + 2**6 + 2**5 + 2**4 + 2**3 + 2**2 + 2**1 + 2**0  or
# 128 + 64 + 32 + 16 + 8 + 4 + 2 + 1 = 255
# but the number 01111111 is
# 0 + 64 + 32 + 16 + 8 + 4 + 2 + 1 = 127
# an example of this in a webpage is https://www.rapidtables.com/convert/number/binary-to-decimal.html?x=01000111
# that converts 01000111 into 71
def binary_string_to_int(binary_string):
    result = 0
    return result


# this needs to convert a number betwen 0-255 into a letter
# the input is a list of numbers and the result should be the message
# the python function that converts a value 0-255 into a letter is chr(the_int)
# with the example above 01000111 = 71 = G
def ascii_list_to_string(ascii_list):
    result = ""
    return result


# read the png file and convert it to list of 0 and 1 where each inner list is a single letter of the message
png_int_list = read_png("code.png")
print("Each character is an inner list of 0 and 1")
print("Each character is represented by 8 0 or 1")
print("That is [0, 1, 0, 0, 0, 1, 1, 1] is a character")
print(png_int_list)

# the characters are easier to read if they are converted to strings '01000111' is an example character in binary
# this step converts each character to a binary string
string_list = int_list_to_string_list(png_int_list)
print()
print("Each character is a string")
print(string_list)

# we want to decode the binary into an integer in the ascii table
# https://asciitable.com/
# once we get this part to work each letter will be a single integer from 0 - 255
# take a look at the ascii table linked above
# this function doesn't work right now because I left it for you
ascii_list = string_list_to_ascii(string_list)
print()
print("Each character is an ascii integer")
print(ascii_list)

# this last step will convert the integers from 0 - 255 into a real character
# after this is done you can read the message
message = ascii_list_to_string(ascii_list)
print()
print("We are done")
print('The message is "' + message + '"')
