
 

import scrapy
 

class AmazonReviewsSpider(scrapy.Spider):
     
    # Spider name
    name = 'amazon_reviews'
     
    
    allowed_domains = ['amazon.in']
     
    
    myBaseUrl = "https://www.amazon.in/JBL-T205BT-Wireless-Earbud-Headphones/product-reviews/B07B9G75Z9/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber="
    start_urls=[]
    
    # Creating list of urls to be scraped by appending page number a the end of base url
    for i in range(1,500):
        start_urls.append(myBaseUrl+str(i))
    
    
    def parse(self, response):
            data = response.css('#cm_cr-review_list')
             
            
            star_rating = data.css('.review-rating')
             
            
            comments = data.css('.review-text')
            count = 0
             
            # Combining the results
            for review in star_rating:
                yield{'stars': ''.join(review.xpath('.//text()').extract()),
                      'comment': ''.join(comments[count].xpath(".//text()").extract())
                     }
                count=count+1
