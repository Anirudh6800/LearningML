class Book:
    
    def __init__(self, title=None, author=None):
        self.title = title
        self.author = author

#     def print_category(self):
#         print("The book author is", self.author)
#         print("The book title is", self.title)

# Book1 = Book("Pro Python", "Marty Alchin")
# Book1.print_category()
# Book2 = Book("Python Programming", "John Zelle")
# Book2.print_category()

class eBook(Book):
    def __init__(self, category, title, author, price, size_in_mb):
        super().__init__(title, author)
        self.category = category
        self.price = price
        self.size_in_mb = size_in_mb
        
    def print_category(self):
        print("The book category is", self.category)
        print("The book price is", self.price)
        print("The book size is", self.size_in_mb)
        
class detatiledBook(eBook):
    def __init__(self, category, title, author, price, size_in_mb, pages):
        super().__init__(category, title , author, price, size_in_mb)
        self.pages = pages
        self.author = author
        
    def print_category(self):
        super().print_category()
        print("The book has", self.pages)
        print("The book publisher is", self.author)



eBook1 = eBook("Programming", "Pro Python", "Marty Alchin", 10, 5)
eBook1.print_category()
    
detailedBook1 = detatiledBook("Programming", "Pro Python", "Marty Alchin", 10, 5, 100)
detailedBook1.print_category()




