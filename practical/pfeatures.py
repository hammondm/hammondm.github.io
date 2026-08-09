from pyfoma import FST

sylplus = FST.re("(uː|eː|oː|ɒ|i|ɔː|æ|u|e|aː|æː|ɛ|ɪ|ɑː|iː|ʊ|ə|a|ɒː|o|ɔ)")

sylminus = FST.re("(w|dː|ɴ|ʒ|m|r|ɹ|d|ʃ|t|j|ɡ|n|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|ɾ|tː|sː|ɣ|z|x|ð|h|t͡ʃʰ|ɡʱ|d͡ʒ|ŋ|β|l|s|f|v|d̪|ɡʷ|ʔ)")

sonplus = FST.re("(w|ɴ|m|r|uː|eː|ɹ|oː|ɒ|i|j|ɔː|n|æ|ɾ|u|e|aː|æː|h|ɛ|ŋ|l|ɪ|ɑː|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

sonminus = FST.re("(dː|ʒ|d|ʃ|t|ɡ|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|tː|sː|ɣ|z|x|ð|t͡ʃʰ|ɡʱ|d͡ʒ|β|s|f|v|d̪|ɡʷ)")

consplus = FST.re("(dː|ɴ|ʒ|m|r|d|ʃ|t|ɡ|n|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|ɾ|tː|sː|ɣ|z|x|ð|h|t͡ʃʰ|ɡʱ|d͡ʒ|ŋ|β|l|s|f|v|d̪|ɡʷ)")

consminus = FST.re("(w|uː|eː|ɹ|oː|ɒ|i|j|ɔː|æ|u|e|aː|æː|ɛ|ɪ|ɑː|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

contplus = FST.re("(w|ʒ|r|uː|eː|ɹ|ʃ|oː|ɒ|i|j|ɔː|æ|xʷ|χ|ʕ|ʃː|ɾ|sː|u|ɣ|z|x|e|aː|ð|æː|h|ɛ|β|l|s|f|ɪ|v|ɑː|iː|ʊ|ə|a|ɒː|o|ɔ)")

contminus = FST.re("(dː|ɴ|m|d|t|ɡ|n|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|t͡ʃ|tː|t͡ʃʰ|ɡʱ|d͡ʒ|ŋ|d̪|ɡʷ|ʔ)")

delrelplus = FST.re("(t͡ʃ|t͡ʃʰ|d͡ʒ)")

delrelunmarked = FST.re("(r|ɾ|ɑː)")

delrelminus = FST.re("(w|dː|ɴ|ʒ|m|uː|eː|ɹ|d|ʃ|oː|ɒ|i|t|j|ɔː|ɡ|n|æ|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|ʕ|ʃː|tː|sː|u|ɣ|z|x|e|aː|ð|æː|h|ɡʱ|ɛ|ŋ|β|l|s|f|ɪ|v|d̪|ɡʷ|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

latplus = FST.re("(l)")

latminus = FST.re("(w|dː|ɴ|ʒ|m|r|uː|eː|ɹ|d|ʃ|oː|ɒ|i|t|j|ɔː|ɡ|n|æ|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|ɾ|tː|sː|u|ɣ|z|x|e|aː|ð|æː|h|t͡ʃʰ|ɡʱ|ɛ|d͡ʒ|ŋ|β|s|f|ɪ|v|d̪|ɡʷ|ɑː|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

nasplus = FST.re("(ɴ|m|n|ŋ)")

nasminus = FST.re("(w|dː|ʒ|r|uː|eː|ɹ|d|ʃ|oː|ɒ|i|t|j|ɔː|ɡ|æ|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|ɾ|tː|sː|u|ɣ|z|x|e|aː|ð|æː|h|t͡ʃʰ|ɡʱ|ɛ|d͡ʒ|β|l|s|f|ɪ|v|d̪|ɡʷ|ɑː|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

stridplus = FST.re("(ʒ|ʃ|χ|t͡ʃ|ʃː|sː|z|t͡ʃʰ|d͡ʒ|s|f|v)")

stridminus = FST.re("(w|dː|ɴ|m|r|uː|eː|ɹ|d|oː|ɒ|i|t|j|ɔː|ɡ|n|æ|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|ʕ|ɾ|tː|u|ɣ|x|e|aː|ð|æː|h|ɡʱ|ɛ|ŋ|β|l|ɪ|d̪|ɡʷ|ɑː|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

voiplus = FST.re("(w|dː|ɴ|ʒ|m|r|uː|eː|ɹ|d|oː|ɒ|i|j|ɔː|ɡ|n|æ|b|ɢ|ʕ|ɾ|u|ɣ|z|e|aː|ð|æː|ɡʱ|ɛ|d͡ʒ|ŋ|β|l|ɪ|v|d̪|ɡʷ|ɑː|iː|ʊ|ə|a|ɒː|o|ɔ)")

voiminus = FST.re("(ʃ|t|kʰ|k|p|tʰ|t̪ʰ|t̪|q|xʷ|χ|t͡ʃ|ʃː|tː|sː|x|h|t͡ʃʰ|s|f|ʔ)")

sgplus = FST.re("(kʰ|tʰ|t̪ʰ|t͡ʃʰ)")

sgminus = FST.re("(w|dː|ɴ|ʒ|m|r|uː|eː|ɹ|d|ʃ|oː|ɒ|i|t|j|ɔː|ɡ|n|æ|k|b|p|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|ɾ|tː|sː|u|ɣ|z|x|e|aː|ð|æː|h|ɡʱ|ɛ|d͡ʒ|ŋ|β|l|s|f|ɪ|v|d̪|ɡʷ|ɑː|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

cgplus = FST.re("(ʔ)")

cgminus = FST.re("(w|dː|ɴ|ʒ|m|r|uː|eː|ɹ|d|ʃ|oː|ɒ|i|t|j|ɔː|ɡ|n|æ|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|ɾ|tː|sː|u|ɣ|z|x|e|aː|ð|æː|h|t͡ʃʰ|ɡʱ|ɛ|d͡ʒ|ŋ|β|l|s|f|ɪ|v|d̪|ɡʷ|ɑː|iː|ʊ|ə|a|ɒː|o|ɔ)")

antplus = FST.re("(dː|m|r|ɹ|d|t|n|b|p|tʰ|t̪ʰ|t̪|ɾ|tː|sː|z|ð|β|l|s|f|v|d̪)")

antunmarked = FST.re("(uː|eː|oː|ɒ|i|ɔː|æ|u|e|aː|æː|ɛ|ɪ|ɑː|iː|ʊ|ə|a|ɒː|o|ɔ)")

antminus = FST.re("(w|ɴ|ʒ|ʃ|j|ɡ|kʰ|k|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|ɣ|x|h|t͡ʃʰ|ɡʱ|d͡ʒ|ŋ|ɡʷ|ʔ)")

corplus = FST.re("(dː|ʒ|r|ɹ|d|ʃ|t|n|tʰ|t̪ʰ|t̪|t͡ʃ|ʃː|ɾ|tː|sː|z|ð|t͡ʃʰ|d͡ʒ|l|s|d̪)")

corminus = FST.re("(w|ɴ|m|uː|eː|oː|ɒ|i|j|ɔː|ɡ|æ|kʰ|k|b|p|q|ɢ|xʷ|χ|ʕ|u|ɣ|x|e|aː|æː|h|ɡʱ|ɛ|ŋ|β|f|ɪ|v|ɡʷ|ɑː|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

distrplus = FST.re("(ʒ|ʃ|t̪ʰ|t̪|t͡ʃ|ʃː|ð|t͡ʃʰ|d͡ʒ|d̪)")

distrunmarked = FST.re("(w|ɴ|m|uː|eː|oː|ɒ|i|j|ɔː|ɡ|æ|kʰ|k|b|p|q|ɢ|xʷ|χ|ʕ|u|ɣ|x|e|aː|æː|h|ɡʱ|ɛ|ŋ|β|f|ɪ|v|ɡʷ|ɑː|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

distrminus = FST.re("(dː|r|ɹ|d|t|n|tʰ|ɾ|tː|sː|z|l|s)")

labplus = FST.re("(w|m|uː|b|p|u|β|f|v)")

labminus = FST.re("(dː|ɴ|ʒ|r|eː|ɹ|d|ʃ|oː|ɒ|i|t|j|ɔː|ɡ|n|æ|kʰ|k|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|ɾ|tː|sː|ɣ|z|x|e|aː|ð|æː|h|t͡ʃʰ|ɡʱ|ɛ|d͡ʒ|ŋ|l|s|ɪ|d̪|ɡʷ|ɑː|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

hiplus = FST.re("(w|uː|ɹ|i|j|ɡ|kʰ|k|xʷ|u|ɣ|x|ɡʱ|ŋ|ɪ|ɡʷ|iː|ʊ)")

hiunmarked = FST.re("(r|ɾ)")

himinus = FST.re("(dː|ɴ|ʒ|m|eː|d|ʃ|oː|ɒ|t|ɔː|n|æ|b|p|tʰ|t̪ʰ|t̪|q|ɢ|χ|t͡ʃ|ʕ|ʃː|tː|sː|z|e|aː|ð|æː|h|t͡ʃʰ|ɛ|d͡ʒ|β|l|s|f|v|d̪|ɑː|ə|a|ʔ|ɒː|o|ɔ)")

loplus = FST.re("(ɒ|æ|ʕ|aː|æː|ɑː|a|ɒː)")

lounmarked = FST.re("(r|ɾ)")

lominus = FST.re("(w|dː|ɴ|ʒ|m|uː|eː|ɹ|d|ʃ|oː|i|t|j|ɔː|ɡ|n|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʃː|tː|sː|u|ɣ|z|x|e|ð|h|t͡ʃʰ|ɡʱ|ɛ|d͡ʒ|ŋ|β|l|s|f|ɪ|v|d̪|ɡʷ|iː|ʊ|ə|ʔ|o|ɔ)")

backplus = FST.re("(w|ɴ|uː|oː|ɒ|ɔː|ɡ|kʰ|k|q|ɢ|xʷ|χ|ʕ|u|ɣ|x|aː|h|ɡʱ|ŋ|ɡʷ|ɑː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

backminus = FST.re("(dː|ʒ|m|r|eː|ɹ|d|ʃ|i|t|j|n|æ|b|p|tʰ|t̪ʰ|t̪|t͡ʃ|ʃː|ɾ|tː|sː|z|e|ð|æː|t͡ʃʰ|ɛ|d͡ʒ|β|l|s|f|ɪ|v|d̪|iː)")

roundplus = FST.re("(w|uː|ɹ|oː|ɒ|ɔː|xʷ|u|ɡʷ|ʊ|ɒː|o|ɔ)")

roundminus = FST.re("(dː|ɴ|ʒ|m|r|eː|d|ʃ|i|t|j|ɡ|n|æ|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|χ|t͡ʃ|ʕ|ʃː|ɾ|tː|sː|ɣ|z|x|e|aː|ð|æː|h|t͡ʃʰ|ɡʱ|ɛ|d͡ʒ|ŋ|β|l|s|f|ɪ|v|d̪|ɑː|iː|ə|a|ʔ)")

velaricminus = FST.re("(w|dː|ɴ|ʒ|m|r|uː|eː|ɹ|d|ʃ|oː|ɒ|i|t|j|ɔː|ɡ|n|æ|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|ɾ|tː|sː|u|ɣ|z|x|e|aː|ð|æː|h|t͡ʃʰ|ɡʱ|ɛ|d͡ʒ|ŋ|β|l|s|f|ɪ|v|d̪|ɡʷ|ɑː|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

tenseplus = FST.re("(uː|eː|oː|ɒ|i|æ|u|e|aː|æː|ɑː|iː|a|ɒː|o)")

tenseunmarked = FST.re("(w|dː|ɴ|ʒ|m|r|ɹ|d|ʃ|t|j|ɡ|n|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|ɾ|tː|sː|ɣ|z|x|ð|h|t͡ʃʰ|ɡʱ|d͡ʒ|ŋ|β|l|s|f|v|d̪|ɡʷ|ʔ)")

tenseminus = FST.re("(ɔː|ɛ|ɪ|ʊ|ə|ɔ)")

longplus = FST.re("(dː|uː|eː|oː|ɔː|ʃː|tː|sː|aː|æː|ɑː|iː|ɒː)")

longminus = FST.re("(w|ɴ|ʒ|m|r|ɹ|d|ʃ|ɒ|i|t|j|ɡ|n|æ|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ɾ|u|ɣ|z|x|e|ð|h|t͡ʃʰ|ɡʱ|ɛ|d͡ʒ|ŋ|β|l|s|f|ɪ|v|d̪|ɡʷ|ʊ|ə|a|ʔ|o|ɔ)")

hitoneunmarked = FST.re("(w|dː|ɴ|ʒ|m|r|uː|eː|ɹ|d|ʃ|oː|ɒ|i|t|j|ɔː|ɡ|n|æ|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|ɾ|tː|sː|u|ɣ|z|x|e|aː|ð|æː|h|t͡ʃʰ|ɡʱ|ɛ|d͡ʒ|ŋ|β|l|s|f|ɪ|v|d̪|ɡʷ|ɑː|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

hiregunmarked = FST.re("(w|dː|ɴ|ʒ|m|r|uː|eː|ɹ|d|ʃ|oː|ɒ|i|t|j|ɔː|ɡ|n|æ|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|ɾ|tː|sː|u|ɣ|z|x|e|aː|ð|æː|h|t͡ʃʰ|ɡʱ|ɛ|d͡ʒ|ŋ|β|l|s|f|ɪ|v|d̪|ɡʷ|ɑː|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

trillplus = FST.re("(r)")

trillunmarked = FST.re("(w|dː|ɴ|ʒ|m|uː|eː|ɹ|d|ʃ|oː|ɒ|i|t|j|ɔː|ɡ|n|æ|kʰ|k|b|p|tʰ|t̪ʰ|t̪|q|ɢ|xʷ|χ|t͡ʃ|ʕ|ʃː|tː|sː|u|ɣ|z|x|e|aː|ð|æː|h|t͡ʃʰ|ɡʱ|ɛ|d͡ʒ|ŋ|β|l|s|f|ɪ|v|d̪|ɡʷ|ɑː|iː|ʊ|ə|a|ʔ|ɒː|o|ɔ)")

trillminus = FST.re("(ɾ)")

