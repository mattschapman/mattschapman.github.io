import scrapy
from mongodb_crawler.items import MongodbCrawlerItem
import time


class TripSpider(scrapy.Spider):
    name = 'trip'
    start_urls = ['https://www.tripadvisor.com/Restaurant_Review-g186361-d23663724-Reviews-The_Gurkha_Palace_The_Chequers_Inn-Oxford_Oxfordshire_England.html', 
                  'https://www.tripadvisor.com/Restaurant_Review-g186361-d14947327-Reviews-Antep_Kitchen-Oxford_Oxfordshire_England.html']
    # with open('np.txt', 'r') as f:
    #     start_urls = [url.strip() for url in f.readlines()]

    def parse(self, response):

        # restaurant_page_url = response.xpath('/html/head/meta[@property="og:url"]/@content').extract()[0]
        restaurant_name = response.css('.fHibz::text').extract_first()

        REVIEW_SELECTOR = './/div[@class="reviewSelector"]'
        for review in response.xpath(REVIEW_SELECTOR):

            item = MongodbCrawlerItem()
            
            DATE_SELECTOR = './/*[@class="ratingDate"]/text()'
            item['review_date'] = review.xpath(DATE_SELECTOR).extract_first()

            VISIT_DATE_SELECTOR = './/div[@class="prw_rup prw_reviews_stay_date_hsx"]/text()'
            item['visit_date'] = review.xpath(VISIT_DATE_SELECTOR).extract_first()

            REVIEWER_SELECTOR = './/div[@class="info_text pointer_cursor"]/div/text()'
            item['reviewer'] = review.xpath(REVIEWER_SELECTOR).extract_first()

            item['restaurant_name'] = restaurant_name

            TITLE_SELECTOR = './/*[@class="noQuotes"]/text()'
            item['title'] = review.xpath(TITLE_SELECTOR).extract_first()
            try:
                item['url'] = 'tripadvisor.com' + review.xpath('.//div[@class="quote"]/a/@href').extract_first()
            except:
                item['url'] = response.xpath('/html/head/meta[@property="og:url"]/@content').extract()[0]
            # item['text'] = review.xpath(TEXT_SELECTOR_ENTRY).extract_first()
            
            TEXT_SELECTOR_PARTIAL = './/*[@class="partial_entry"]/text()'
            TEXT_SELECTOR_MORE = './/*[@class="postSnippet"]/text()'
            text_partial = review.xpath(TEXT_SELECTOR_PARTIAL).extract_first()
            text_more = review.xpath(TEXT_SELECTOR_MORE).extract_first()
            if text_more:
                item['text'] = text_partial[:-3] + ' ' + text_more 
            else:
                item['text'] = text_partial

            yield item

            time.sleep(0.25)

            # This one is when you just want to print out the output    
            # yield {
            #   'title': review.xpath(TITLE_SELECTOR).extract_first(),
            #   'text_partial': review.xpath(TEXT_SELECTOR_PARTIAL).extract_first(),
            #   'text_more': review.xpath(TEXT_SELECTOR_MORE).extract_first(),
            #   'date': review.xpath(DATE_SELECTOR).extract_first(),
            # }

        NEXT_PAGE_SELECTOR = './/*[@class="nav next ui_button primary"]/@href'
        next_page = response.xpath(NEXT_PAGE_SELECTOR).extract_first()
        if next_page:
            yield scrapy.Request(
                    # response.urljoin(next_page),
                    url=response.urljoin(next_page),
                    callback=self.parse 
                )