import html.parser
class HTMLFilter(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = ''
    def handle_data(self, data):
        self.text += data

parser = HTMLFilter()
with open('lung_cancer_prediction_final.html', encoding='utf-8') as f:
    parser.feed(f.read())
with open('extracted_code.txt', 'w', encoding='utf-8') as f:
    f.write(parser.text)
