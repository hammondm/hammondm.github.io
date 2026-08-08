from pyfoma import FST

sylplus = FST.re("(iː|ɨ|l̩|ɨː|u|i|eː|ɛ|aː|ə|o|ɔ|uː|ɪ|e|r̩|ɑː|ʊ|oː|a)")

sylminus = FST.re("(d|ð|r̥|j|χ|ɹ|i̯|k|n̥|ɨ̯|ʉ̯|n|θ|h|s|ŋ|ɬ|f|ŋ̊|ʃ|d͡ʒ|t͡ʃ|a̯|ɪ̯|x|z|v|w|ʒ|m|r|l|u̯|b|ɡ|p|t|m̥)")

sonplus = FST.re("(iː|ɨ|l̩|ɨː|r̥|j|u|ɹ|i̯|i|n̥|ɨ̯|ʉ̯|n|h|ŋ|eː|ɛ|ŋ̊|a̯|aː|ɪ̯|w|ə|o|m|r|ɔ|uː|l|ɪ|u̯|e|r̩|ɑː|ʊ|oː|m̥|a)")

sonminus = FST.re("(d|ð|χ|k|θ|s|ɬ|f|ʃ|d͡ʒ|t͡ʃ|x|z|v|ʒ|b|ɡ|p|t)")

consplus = FST.re("(d|ð|l̩|r̥|χ|k|n̥|n|θ|h|s|ŋ|ɬ|f|ŋ̊|ʃ|d͡ʒ|t͡ʃ|x|z|v|ʒ|m|r|l|b|r̩|ɡ|p|t|m̥)")

consminus = FST.re("(iː|ɨ|ɨː|j|u|ɹ|i̯|i|ɨ̯|ʉ̯|eː|ɛ|a̯|aː|ɪ̯|w|ə|o|ɔ|uː|ɪ|u̯|e|ɑː|ʊ|oː|a)")

contplus = FST.re("(iː|ð|ɨ|l̩|ɨː|r̥|j|χ|u|ɹ|i̯|i|ɨ̯|ʉ̯|θ|h|s|eː|ɛ|ɬ|f|ʃ|a̯|aː|ɪ̯|x|z|v|w|ʒ|ə|o|r|ɔ|uː|l|ɪ|u̯|e|r̩|ɑː|ʊ|oː|a)")

contminus = FST.re("(d|k|n̥|n|ŋ|ŋ̊|d͡ʒ|t͡ʃ|m|b|ɡ|p|t|m̥)")

delrelplus = FST.re("(ɬ|d͡ʒ|t͡ʃ)")

delrelunmarked = FST.re("(r̥|ʉ̯|r|r̩|ɑː)")

delrelminus = FST.re("(iː|d|ð|ɨ|l̩|ɨː|j|χ|u|ɹ|i̯|k|i|n̥|ɨ̯|n|θ|h|s|ŋ|eː|ɛ|f|ŋ̊|ʃ|a̯|aː|ɪ̯|x|z|v|w|ʒ|ə|o|m|ɔ|uː|l|ɪ|u̯|b|e|ʊ|ɡ|oː|p|t|m̥|a)")

latplus = FST.re("(l̩|ɬ|l)")

latminus = FST.re("(iː|d|ð|ɨ|ɨː|r̥|j|χ|u|ɹ|i̯|k|i|n̥|ɨ̯|ʉ̯|n|θ|h|s|ŋ|eː|ɛ|f|ŋ̊|ʃ|d͡ʒ|t͡ʃ|a̯|aː|ɪ̯|x|z|v|w|ʒ|ə|o|m|r|ɔ|uː|ɪ|u̯|b|e|r̩|ɑː|ʊ|ɡ|oː|p|t|m̥|a)")

nasplus = FST.re("(n̥|n|ŋ|ŋ̊|m|m̥)")

nasminus = FST.re("(iː|d|ð|ɨ|l̩|ɨː|r̥|j|χ|u|ɹ|i̯|k|i|ɨ̯|ʉ̯|θ|h|s|eː|ɛ|ɬ|f|ʃ|d͡ʒ|t͡ʃ|a̯|aː|ɪ̯|x|z|v|w|ʒ|ə|o|r|ɔ|uː|l|ɪ|u̯|b|e|r̩|ɑː|ʊ|ɡ|oː|p|t|a)")

stridplus = FST.re("(χ|s|f|ʃ|d͡ʒ|t͡ʃ|z|v|ʒ)")

stridminus = FST.re("(iː|d|ð|ɨ|l̩|ɨː|r̥|j|u|ɹ|i̯|k|i|n̥|ɨ̯|ʉ̯|n|θ|h|ŋ|eː|ɛ|ɬ|ŋ̊|a̯|aː|ɪ̯|x|w|ə|o|m|r|ɔ|uː|l|ɪ|u̯|b|e|r̩|ɑː|ʊ|ɡ|oː|p|t|m̥|a)")

voiplus = FST.re("(iː|d|ð|ɨ|l̩|ɨː|j|u|ɹ|i̯|i|ɨ̯|ʉ̯|n|ŋ|eː|ɛ|d͡ʒ|a̯|aː|ɪ̯|z|v|w|ʒ|ə|o|m|r|ɔ|uː|l|ɪ|u̯|b|e|r̩|ɑː|ʊ|ɡ|oː|a)")

voiminus = FST.re("(r̥|χ|k|n̥|θ|h|s|ɬ|f|ŋ̊|ʃ|t͡ʃ|x|p|t|m̥)")

sgminus = FST.re("(iː|d|ð|ɨ|l̩|ɨː|r̥|j|χ|u|ɹ|i̯|k|i|n̥|ɨ̯|ʉ̯|n|θ|h|s|ŋ|eː|ɛ|ɬ|f|ŋ̊|ʃ|d͡ʒ|t͡ʃ|a̯|aː|ɪ̯|x|z|v|w|ʒ|ə|o|m|r|ɔ|uː|l|ɪ|u̯|b|e|r̩|ɑː|ʊ|ɡ|oː|p|t|m̥|a)")

cgminus = FST.re("(iː|d|ð|ɨ|l̩|ɨː|r̥|j|χ|u|ɹ|i̯|k|i|n̥|ɨ̯|ʉ̯|n|θ|h|s|ŋ|eː|ɛ|ɬ|f|ŋ̊|ʃ|d͡ʒ|t͡ʃ|a̯|aː|ɪ̯|x|z|v|w|ʒ|ə|o|m|r|ɔ|uː|l|ɪ|u̯|b|e|r̩|ɑː|ʊ|ɡ|oː|p|t|m̥|a)")

antplus = FST.re("(d|ð|l̩|r̥|ɹ|n̥|n|θ|s|ɬ|f|z|v|m|r|l|b|r̩|p|t|m̥)")

antunmarked = FST.re("(iː|ɨ|ɨː|u|i̯|i|ɨ̯|ʉ̯|eː|ɛ|a̯|aː|ɪ̯|ə|o|ɔ|uː|ɪ|u̯|e|ɑː|ʊ|oː|a)")

antminus = FST.re("(j|χ|k|h|ŋ|ŋ̊|ʃ|d͡ʒ|t͡ʃ|x|w|ʒ|ɡ)")

corplus = FST.re("(d|ð|l̩|r̥|ɹ|n̥|n|θ|s|ɬ|ʃ|d͡ʒ|t͡ʃ|z|ʒ|r|l|r̩|t)")

corminus = FST.re("(iː|ɨ|ɨː|j|χ|u|i̯|k|i|ɨ̯|ʉ̯|h|ŋ|eː|ɛ|f|ŋ̊|a̯|aː|ɪ̯|x|v|w|ə|o|m|ɔ|uː|ɪ|u̯|b|e|ɑː|ʊ|ɡ|oː|p|m̥|a)")

distrplus = FST.re("(ð|θ|ʃ|d͡ʒ|t͡ʃ|ʒ)")

distrunmarked = FST.re("(iː|ɨ|ɨː|j|χ|u|i̯|k|i|ɨ̯|ʉ̯|h|ŋ|eː|ɛ|f|ŋ̊|a̯|aː|ɪ̯|x|v|w|ə|o|m|ɔ|uː|ɪ|u̯|b|e|ɑː|ʊ|ɡ|oː|p|m̥|a)")

distrminus = FST.re("(d|l̩|r̥|ɹ|n̥|n|s|ɬ|z|r|l|r̩|t)")

labplus = FST.re("(u|ʉ̯|f|v|w|m|uː|u̯|b|p|m̥)")

labminus = FST.re("(iː|d|ð|ɨ|l̩|ɨː|r̥|j|χ|ɹ|i̯|k|i|n̥|ɨ̯|n|θ|h|s|ŋ|eː|ɛ|ɬ|ŋ̊|ʃ|d͡ʒ|t͡ʃ|a̯|aː|ɪ̯|x|z|ʒ|ə|o|r|ɔ|l|ɪ|e|r̩|ɑː|ʊ|ɡ|oː|t|a)")

hiplus = FST.re("(iː|ɨ|ɨː|j|u|ɹ|i̯|k|i|ɨ̯|ʉ̯|ŋ|ŋ̊|ɪ̯|x|w|uː|ɪ|u̯|ʊ|ɡ)")

hiunmarked = FST.re("(r̥|ɬ|r|r̩)")

himinus = FST.re("(d|ð|l̩|χ|n̥|n|θ|h|s|eː|ɛ|f|ʃ|d͡ʒ|t͡ʃ|a̯|aː|z|v|ʒ|ə|o|m|ɔ|l|b|e|ɑː|oː|p|t|m̥|a)")

loplus = FST.re("(a̯|aː|ɑː|a)")

lounmarked = FST.re("(r̥|ɬ|r|r̩)")

lominus = FST.re("(iː|d|ð|ɨ|l̩|ɨː|j|χ|u|ɹ|i̯|k|i|n̥|ɨ̯|ʉ̯|n|θ|h|s|ŋ|eː|ɛ|f|ŋ̊|ʃ|d͡ʒ|t͡ʃ|ɪ̯|x|z|v|w|ʒ|ə|o|m|ɔ|uː|l|ɪ|u̯|b|e|ʊ|ɡ|oː|p|t|m̥)")

backplus = FST.re("(ɨ|ɨː|χ|u|k|ɨ̯|ʉ̯|h|ŋ|ŋ̊|a̯|aː|x|w|ə|o|ɔ|uː|u̯|ɑː|ʊ|ɡ|oː|a)")

backminus = FST.re("(iː|d|ð|l̩|r̥|j|ɹ|i̯|i|n̥|n|θ|s|eː|ɛ|ɬ|f|ʃ|d͡ʒ|t͡ʃ|ɪ̯|z|v|ʒ|m|r|l|ɪ|b|e|r̩|p|t|m̥)")

roundplus = FST.re("(u|ɹ|ʉ̯|w|o|ɔ|uː|u̯|ʊ|oː)")

roundminus = FST.re("(iː|d|ð|ɨ|l̩|ɨː|r̥|j|χ|i̯|k|i|n̥|ɨ̯|n|θ|h|s|ŋ|eː|ɛ|ɬ|f|ŋ̊|ʃ|d͡ʒ|t͡ʃ|a̯|aː|ɪ̯|x|z|v|ʒ|ə|m|r|l|ɪ|b|e|r̩|ɑː|ɡ|p|t|m̥|a)")

velaricminus = FST.re("(iː|d|ð|ɨ|l̩|ɨː|r̥|j|χ|u|ɹ|i̯|k|i|n̥|ɨ̯|ʉ̯|n|θ|h|s|ŋ|eː|ɛ|ɬ|f|ŋ̊|ʃ|d͡ʒ|t͡ʃ|a̯|aː|ɪ̯|x|z|v|w|ʒ|ə|o|m|r|ɔ|uː|l|ɪ|u̯|b|e|r̩|ɑː|ʊ|ɡ|oː|p|t|m̥|a)")

tenseplus = FST.re("(iː|ɨ|ɨː|u|i̯|i|ɨ̯|ʉ̯|eː|a̯|aː|o|uː|u̯|e|ɑː|oː|a)")

tenseunmarked = FST.re("(d|ð|l̩|r̥|j|χ|ɹ|k|n̥|n|θ|h|s|ŋ|ɬ|f|ŋ̊|ʃ|d͡ʒ|t͡ʃ|x|z|v|w|ʒ|m|r|l|b|r̩|ɡ|p|t|m̥)")

tenseminus = FST.re("(ɛ|ɪ̯|ə|ɔ|ɪ|ʊ)")

longplus = FST.re("(iː|ɨː|eː|aː|uː|ɑː|oː)")

longminus = FST.re("(d|ð|ɨ|l̩|r̥|j|χ|u|ɹ|i̯|k|i|n̥|ɨ̯|ʉ̯|n|θ|h|s|ŋ|ɛ|ɬ|f|ŋ̊|ʃ|d͡ʒ|t͡ʃ|a̯|ɪ̯|x|z|v|w|ʒ|ə|o|m|r|ɔ|l|ɪ|u̯|b|e|r̩|ʊ|ɡ|p|t|m̥|a)")

hitoneunmarked = FST.re("(iː|d|ð|ɨ|l̩|ɨː|r̥|j|χ|u|ɹ|i̯|k|i|n̥|ɨ̯|ʉ̯|n|θ|h|s|ŋ|eː|ɛ|ɬ|f|ŋ̊|ʃ|d͡ʒ|t͡ʃ|a̯|aː|ɪ̯|x|z|v|w|ʒ|ə|o|m|r|ɔ|uː|l|ɪ|u̯|b|e|r̩|ɑː|ʊ|ɡ|oː|p|t|m̥|a)")

hiregunmarked = FST.re("(iː|d|ð|ɨ|l̩|ɨː|r̥|j|χ|u|ɹ|i̯|k|i|n̥|ɨ̯|ʉ̯|n|θ|h|s|ŋ|eː|ɛ|ɬ|f|ŋ̊|ʃ|d͡ʒ|t͡ʃ|a̯|aː|ɪ̯|x|z|v|w|ʒ|ə|o|m|r|ɔ|uː|l|ɪ|u̯|b|e|r̩|ɑː|ʊ|ɡ|oː|p|t|m̥|a)")

