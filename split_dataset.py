# split dataset into train, val & test (70, 10 & 10 images / class)
import splitfolders
splitfolders.fixed(r'C:\Users\Jakob\Downloads\wikiart\wikiart', 
                   output=r'C:\Users\Jakob\Downloads\wikiart\wikiart_ASaAI_test', 
                   seed=1337, 
                   fixed=(10,10,70), 
                   oversample=False,
                   group_prefix=None, 
                   move=False)