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


png_contents = read_png("code.png")
print(png_contents)
