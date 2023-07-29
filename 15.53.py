#I FUCKING IDF, ISRAEL SHALL BURN TO THE GROUND, W PALESTINE FUCK YOU ISRAEL!!11`1

#objective -> idf and tfid i think

#Libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text


if __name__ == "__main__":    
    quotes = ["A horse, a horse, my kingdom for a horse!",
            "Horse sense is the thing a horse has which keeps it from betting on people."
            "I’ve often said there is nothing better for the inside of the man, than the outside of the horse.",
            "A man on a horse is spiritually, as well as physically, bigger then a man on foot.",
            "No heaven can heaven be, if my horse isn’t there to welcome me."]
    
    #turn to vector blablabla
    vectorizer= text.CountVectorizer(stop_words=list(text.ENGLISH_STOP_WORDS))
    vectorizer.fit(quotes)
    token=vectorizer.transform(quotes)
    tokenarr=token.toarray()
    
    #show words blabla cuz why not
    df=pd.DataFrame(tokenarr, columns=vectorizer.get_feature_names_out())
    print(df)
    
    #get tfid model and fit the model
    tfidf=text.TfidfTransformer()
    tfidf.fit(token)
    print((pd.DataFrame({"word": vectorizer.get_feature_names_out(),
                        "frequency": tfidf.idf_})).sort_values(by="frequency"))