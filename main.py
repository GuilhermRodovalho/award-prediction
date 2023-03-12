#main.py

from requests_html import HTMLSession
from sqlalchemy import create_engine

def get_reviews(sess, url):    
    reviews_list = {"reviews":[], 'url':url}
    res = sess.get(url)
    html = res.html
    reviews = html.find('.user_reviews', first=True).find('.review')
    count = 0
    for review in reviews:
        count += 1
        print(review.find(".date", first=True).text)
    next_page = html.find('.flipper.next', first=True)
    if next_page:
        try:
            next_page_url = next_page.absolute_links.pop() 
            reviews_list['reviews'].extend(get_reviews(next_page_url)['reviews'])
        except KeyError:
            print(KeyError)
    print(count)
    return reviews_list

url = 'https://www.metacritic.com/movie/whiplash/user-reviews?sort-by=date&num_items=100'





if __name__ == '__main__':
    sess = HTMLSession()
    db = create_engine('postgresql://postgres:123456@localhost:5445/tcc', echo = True)
    result_set = db.execute("SELECT * FROM filme")  
    for r in result_set:  
        print(r['nome'])