import png


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


def int_list_to_string_list(int_list):
    result = []
    for letter_list in int_list:
        letter_binary_string = ""
        for binary_digit in letter_list:
            letter_binary_string = letter_binary_string + str(binary_digit)
        result.append(letter_binary_string)
    return result


def string_list_to_ascii(string_list):
    result = []
    for letter_as_binary_string in string_list:
        result.append(binary_string_to_int(letter_as_binary_string))
    return result


# The code above this works fine. you can just ignore it if you like.
# However, the code below needs to be filled in to get a result


def binary_string_to_int(binary_string):
    result = 0
    return result


def ascii_list_to_string(ascii_list):
    result = ""
    return result


png_int_list = read_png("code.png")
string_list = int_list_to_string_list(png_int_list)
print("Each character is a string")
print(string_list)
ascii_list = string_list_to_ascii(string_list)
print()
print("Each character is an ascii integer")
print(ascii_list)
message = ascii_list_to_string(ascii_list)
print()
print("We are done")
print('The message is "' + message + '"')
