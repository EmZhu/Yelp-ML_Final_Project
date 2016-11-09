

#Predicting Yelp Review Ratings Using Anchors
Zi Gu, Erika Lage, Jielei Zhu


#1	Introduction
					
Yelp’s Dataset Challenge includes several suggestions for topics of research on its website [7]. One of the suggestions under Natural Language Processing is: “How well can you guess a review's rating from its text alone?” As a standalone problem, this is not particularly interesting since text reviews are always posted with an accompanying star rating. But if we are able to find a low dimensional representation of the reviews that we can use to perform this kind of prediction with an acceptable amount of accuracy, the new representation can be used to build a recommendation system since we assume that it effectively approximates the dimensions that users care about when rating restaurants.


Review texts cover a huge mixture of topics ranging from food to service, and different users weight these aspects differently. Our goal is to pick out the important aspects of restaurant reviews and to choose a set of anchors––or terms that represent these aspects––that allow us to predict the star rating of a review with a comparable amount of accuracy to using the full bag of words representation of the text.
					
#2	Related Work
				
Several other groups have tackled the problem of dimensionality reduction in Yelp reviews.  [2] and [3] use modifications on LDA topic modelling to approximate the dimensions that users rate restaurants along.  Both groups are concerned with making their automatically-generated topics more human readable, since this produces a better approximation of what people care about in restaurants.


Anchors are a method for extracting latent dimensions of a text marked as important by experts [6].  They generate a lower dimensional representation of the text that can be used as features in a classifier similar to LDA generated topics.  The low-dimensional anchor representation has the advantage of encoding expert knowledge into the model, which should produce better results than automatically-generated topics that will not resemble the dimensions that users rate restaurants along as closely.
					
#3	Problem Definition and Algorithm 


#3.1 Task


Our task is to find the most important topics addressed in restaurant reviews and to use them to create a low-dimensional representation of the reviews by finding corresponding anchors.  The value of each feature in these low-dimensional topic feature vectors indicates the likelihood of that specific topic appearing in the text.  Our aim is to classify the reviews as positive or negative using the anchor representation and to obtain comparable accuracy to using a bag of words representation.
		
			
#3.2 Algorithm 


Anchors are terms that correspond to latent variables and that allow us to infer their value from the text [6].  In order to be an anchor, a term must only occur in instances where the latent variable it represents takes a positive value.  In other words, if we see the anchor in the text, we know that the latent variable it represents is positive.  If we do not observe the anchor, we do not know anything about the latent variable it represents.  A good anchor should also be conditionally independent from the other words in the text.  This is described in more depth in [6].


Choosing Anchors
One of the advantages of using anchors is that they allow us to encode a certain amount of expert knowledge into the model.  To find anchors, we came up with a list of questions that an expert in the field would need to know the answer to in order to give an informed opinion.  Put differently, what would a Mexican restaurant connoisseur need to know to accurately perform our classification task?  After generating a list of 10 to 20 of these questions, we chose candidate anchors so that when we saw the term in the text, regardless of what other terms appeared, one of our questions was unambiguously answered.  


As an example, the highest-weighted anchor in the linear classifier trained on the low dimensional anchor representations was “rude”.  The question it corresponds to is “were the people who worked at the restaurant polite?” and the answer is “no.”  “Rude” is a good anchor, because when we see it in a review, we do not need to read the rest of the review to guess that the reviewer thought the staff was not polite, since it is rarely used in the negative (i.e. a reviewer saying that a polite waiter was “not rude”).  It is possible that the reviewer could be saying something negative about another customer, but this makes up only a small minority of cases.  In practice, few words absolutely meet the criteria to be an anchor, so we chose words that met the criteria as closely as possible.


We selected candidate anchors from lists of frequent words by topic generated from topic modeling, and lists of bigrams that appear more frequently in positive reviews than negative ones or vice versa.  We narrowed down our list by selecting terms that occur frequently enough in the reviews to be useful in classification, and that skew towards appearing in one class or the other.  We used a cutoff of around 100 instances in the training set since it was hard to guarantee that there would be examples to train on with less than that.  Once we had generated the final list, we condensed terms that answered the same question in the same way into single, multi-term anchors since they imply the same value of a latent variable.  Our final list contained 20 anchors, some of which concatenated together multiple terms to represent the same latent variable (such as “authentic” and “traditional” which are synonyms).  See the full list of anchors in Appendix A.


Labelling Reviews with Anchor Probabilities
We implemented the algorithm described in [6] for estimating the probabilities of latent variables corresponding to anchors, which uses the procedure for learning positivistic labels.  On a high level, this involves splitting the data into a train and a validate set, then training a logistic regression model on the training set using the presence of the anchor as the label.  The next step is to compute the average of the probabilities generated for reviews in the validate set where the anchor occurs, which is used to calibrate the model by dividing the probability generated by the regression.  See [6] for more details.
					
#4	Experimental Evaluation 


#4.1 Data


We used restaurant reviews from the Yelp Dataset Challenge, which contains about 2.2 million reviews for 77,000 businesses across 10 cities [7]. The data includes review text, star rating, and a business identification number (used in place of business name) for each review, along with other fields that we did not use.  From these, we selected about 30,000 reviews of Mexican restaurants in Phoenix, Arizona.  Using this subset of reviews allowed us to quickly train and tune our system, but the expectation is that the results generalize to Mexican restaurants in other cities.  This was true when we ran our system on Mexican restaurants in Las Vegas, which produced very similar results.


The star ratings in the Yelp dataset ranged from 1 to 5 with 1 being the most negative, and 5 the most positive. Instead of treating each individual star rating as a separate class, we converted them into two classes –– positive and negative. The positive class consisted of reviews that gave 4 or 5 star ratings, and the negative class consisted of reviews that gave 1, 2 or 3 star ratings. The rationale for this split is that positive and negative classes will be more robust to individual variations in the meaning of 4 vs. 5 stars, or 1 vs. 2 stars.		
					
#4.2 Methodology


We split the dataset into 80% development (training + validation) and 20% test. Within the development set, 80% was used for training and 20% for validation. For each review in our dataset, we stripped off irrelevant data fields(e.g. user id) and kept only review text and star ratings, which translate into the X and Y in our supervised learning model. 


We implemented two models: an anchor model and a bag-of-words model. The bag-of-words model served as a benchmark. 


For the bag of words model, we first converted the review texts into feature vectors of length k, with k being the number of unique words in the dataset that occurred in more than 50 reviews. Each element of the feature vector is either 0 or 1––1 if the word appears in the review text, 0 if it does not. For classification and prediction, we used the off-the-shelf Support Vector Classification(SVC) function from Scikit-learn [5]. 


We initially attempted to mark the scope of negation when constructing our bag of words feature representation, but this split some of the terms we chose as anchors into two terms, neither of which occurred frequently enough to meet our criteria.  We decided not to mark negation for this reason.


The function we used for topic modelling was the Latent Dirichlet Allocation module from Gensim [4]. What we found was that the words from 20-topic lists were most helpful –– they were nuanced enough without being too detailed.


In order to preserve bigrams chosen as anchors in the bag of words representation, we underscored two-word anchors in the text before generating the bag of words feature vectors.


In an attempt to maximize conditional independence between anchors and the rest of the text, we tried getting rid of words that appeared in 1, 3 and 5 word windows around each anchor.  We used a 1 word window in our final system.


We tried a few classification functions from Scikit-learn for the anchor features and found that random forest achieved the highest accuracy on the validation set with 80 estimators, a minimum of 10 samples per leaf, and a maximum depth of 30 [5].  This was used to calculate our final result.


To evaluate the anchor model, we compared the results to the benchmark accuracy (bag of words), and a baseline accuracy which involved predicting positive labels for all reviews.  The ratio of positive to negative reviews is around 2 to 1, which gives us a baseline of around 68% instead of 50% for an evenly split dataset.
		
#4.3 Results
		
The benchmark accuracy with the bag of words model was 82%. Our anchor model, on the other hand, achieved 80.6% accuracy. The baseline accuracy was calculated by predicting all reviews as positive, which gave 69.2% accuracy. 


Because our task is a binary classification problem, we also compared the AUC(area under the ROC curve) of the models to better understand the differences in performance(Figure 1).  



Figure 1: Receiver Operating Characteristic(ROC) curve for anchor model and baseline.


For classification accuracy, the bag of words model did the best, then the anchor model, and lastly the baseline model, but the difference in performance between anchors and bag of words was insignificant compared to the difference to the baseline.   
				
#4.4 Discussion


With an insignificant difference in prediction accuracy between the bag of words model and the anchor model, we are confident that we have found a lower dimensional representation of restaurant reviews that preserves the full meaning of the text.
					
#5	Conclusions
	
As mentioned above, we were able to find a set of anchors that corresponded to a much lower dimensional representation of the reviews that maintained almost the same accuracy in predicting star ratings. This suggests that we can use these anchors and the latent variables they represent to build out a recommendation system that depends on the important features of the restuarants.


Learning the probabilities of each anchor occurring in the text is an unsupervised process, so there is no easy way to score it besides the performance of the anchor feature vectors in the final model.  We tried to do a high level qualitative analysis of how well our model predicts the latent quantities we hope to represent, but manually labelling a significant amount of data was too time consuming to get interpretable results.  From what we did have time to do, it seems like some latent variables are being predicted with better accuracy than others, and understanding which latent variables can be predicted well would help us to choose better anchors.  In future work, we would like to manually label a subset of reviews with the values of the latent variables represented by the anchors, and score the anchor predictions made by our system against these labels.  This would help us to fine tune the process of choosing anchors.


#To Run

1) Yelp_Project.py to pull out selected reviews 2) partition_review.py to partition the reviews into test, train, validate files 3) preprocessing.py to generate the dictionary and the files of feature vectors for best, train, validate 4) classify.py to run an svm classifier for baseline accuracy and to pull out top weights of L1 penalty classifier to use in selecting anchors


Bibliography
					
1. Elkan C, Noto K. Learning classifiers from only positive and unlabeled data. In: KDD; 
2008. p. 213–220.                         	
2. Julian McAuley and Jure Leskovec. “Hidden factors and hidden topics: understanding 
rating dimensions with review text.” In Proceedings of the 7th ACM conference 
on Recommender systems, pages 165–172. ACM. 2013.
3. Linshi, Jack. "Personalizing Yelp Star Ratings: a Semantic Topic Modeling Approach." 
Yale University, 2014.
4. Rehurek, R., & Sojka, P. (2010). Software framework for topic modelling with large 
corpora. LREC.
 5. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 
2011.
6. Y. Halpern, Y.D. Choi, S. Horng, D. Sontag. “Using Anchors to Estimate Clinical State 
without Labeled Data.” American Medical Informatics Association (AMIA) Annual Symposium, Nov. 2014.
7. Yelp. Open Dataset Challenge Dataset. Retrieved from 
https://www.yelp.com/dataset_challenge.


























Appendix A


Anchors


overpriced/over priced
complimentary


happy hour


spicy/flavorful/hot


tender/juicy
bland
stale/soggy/canned 
best tacos
gourmet
fresh/homemade


decor/ambience
laid back/casual


date


bright


hidden gem


rude


took forever


line


authentic/traditional


breakfast/brunch




           










































      Appendix B


Weights of a Linear Classifier Trained on Anchor Features (sorted by absolute value)


rude                         
-2.18450411141
bland  
-1.90822036573
stale / soggy / canned       
-1.67896766475
best tacos / best taco        
0.746970015735
hidden gem / little gem       
0.55529937157
gourmet                      
0.391508431384
laid back / casual           
0.335325930495
line                         
-0.259653343036
took forever / finally came   
-0.179635125193
breakfast / brunch           
0.174904741554
tender / juicy / fresh       
0.170117679717
fresh / homemade              
0.170086102241
spicy / flavorful / hot       
0.149046300368
authentic / traditional       
0.116502286179
date                          
0.0949747814786
bright                        
0.0856934643865
decor / ambience              
-0.0411095141565
complimentary                 
-0.0299134173598
happy hour                    
-0.0149679931078


