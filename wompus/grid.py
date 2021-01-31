import arcade

class Grid():
    def __init__(self, width, height, columns, rows):
        self.width = width
        self.height = height
        self.columns = columns
        self.rows = rows
        self.column_width = width / self.columns
        self.row_height = height / self.rows

    def Draw(self):
        for i in range(self.columns - 1):
            x = (i + 1) * self.column_width
            arcade.draw_line(x, 0, x, self.height, arcade.color.BLACK, 2)

        for j in range(self.rows - 1):
            y = (j + 1) * self.row_height
            arcade.draw_line(0, y, self.width, y, arcade.color.BLACK, 2)