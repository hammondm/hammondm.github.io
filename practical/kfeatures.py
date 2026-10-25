from pyfoma import FST

sylplus = FST.re("(ɘ|y|ɑ|œ|ɛ|o|oː|u|eː|ʏ|aː|øː|i|ɔ|uː|e|ø|yː|a|ɯ|ɑː)")

sylminus = FST.re("(ɟ|k|ɢ|l|c|ʂ|z|n|x|m|j|ʒ|ɫ|tː|b|f|ɾ|tʲ|ʃ|v|mʲ|t͡s|w|r|t|s|ɡ|ɣ|sː|ʁ|t͡ʃ|p|β|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|χ|d|qː)")

sonplus = FST.re("(ɘ|y|l|ɑ|n|m|j|œ|ɫ|ɛ|o|oː|ɾ|u|eː|mʲ|w|r|ʏ|aː|øː|i|ɔ|uː|e|h|ŋ|β̞|ø|yː|a|ɯ|ɑː)")

sonminus = FST.re("(ɟ|k|ɢ|c|ʂ|z|x|ʒ|tː|b|f|tʲ|ʃ|v|t͡s|t|s|ɡ|ɣ|sː|ʁ|t͡ʃ|p|β|t͡ɕ|q|d͡ʒ|χ|d|qː)")

consplus = FST.re("(ɟ|k|ɢ|l|c|ʂ|z|n|x|m|ʒ|ɫ|tː|b|f|ɾ|tʲ|ʃ|v|mʲ|t͡s|r|t|s|ɡ|ɣ|sː|ʁ|t͡ʃ|p|β|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|χ|d|qː)")

consminus = FST.re("(ɘ|y|ɑ|j|œ|ɛ|o|oː|u|eː|w|ʏ|aː|øː|i|ɔ|uː|e|ø|yː|a|ɯ|ɑː)")

contplus = FST.re("(ɘ|y|l|ʂ|ɑ|z|x|j|œ|ʒ|ɫ|ɛ|o|f|oː|ɾ|u|ʃ|eː|v|w|r|ʏ|s|ɣ|sː|aː|øː|ʁ|i|ɔ|uː|β|e|h|ø|yː|χ|a|ɯ|ɑː)")

contminus = FST.re("(ɟ|k|ɢ|c|n|m|tː|b|tʲ|mʲ|t͡s|t|ɡ|t͡ʃ|p|t͡ɕ|ŋ|q|β̞|d͡ʒ|d|qː)")

delrelplus = FST.re("(t͡s|t͡ʃ|t͡ɕ|d͡ʒ)")

delrelunmarked = FST.re("(ɘ|ɑ|ɫ|ɾ|r|ɑː)")

delrelminus = FST.re("(ɟ|k|y|ɢ|l|c|ʂ|z|n|x|m|j|œ|ʒ|ɛ|o|tː|b|f|oː|u|tʲ|ʃ|eː|v|mʲ|w|ʏ|t|s|ɡ|ɣ|sː|aː|øː|ʁ|i|ɔ|p|uː|β|e|h|ŋ|q|β̞|ø|yː|χ|a|d|ɯ|qː)")

latplus = FST.re("(l|ɫ)")

latminus = FST.re("(ɟ|ɘ|k|y|ɢ|c|ʂ|ɑ|z|n|x|m|j|œ|ʒ|ɛ|o|tː|b|f|oː|ɾ|u|tʲ|ʃ|eː|v|mʲ|t͡s|w|r|ʏ|t|s|ɡ|ɣ|sː|aː|øː|ʁ|i|ɔ|t͡ʃ|p|uː|β|e|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|ø|yː|χ|a|d|ɯ|qː|ɑː)")

nasplus = FST.re("(n|m|mʲ|ŋ|β̞)")

nasminus = FST.re("(ɟ|ɘ|k|y|ɢ|l|c|ʂ|ɑ|z|x|j|œ|ʒ|ɫ|ɛ|o|tː|b|f|oː|ɾ|u|tʲ|ʃ|eː|v|t͡s|w|r|ʏ|t|s|ɡ|ɣ|sː|aː|øː|ʁ|i|ɔ|t͡ʃ|p|uː|β|e|h|t͡ɕ|q|d͡ʒ|ø|yː|χ|a|d|ɯ|qː|ɑː)")

stridplus = FST.re("(ʂ|z|ʒ|f|ʃ|v|t͡s|s|sː|ʁ|t͡ʃ|t͡ɕ|d͡ʒ|χ)")

stridminus = FST.re("(ɟ|ɘ|k|y|ɢ|l|c|ɑ|n|x|m|j|œ|ɫ|ɛ|o|tː|b|oː|ɾ|u|tʲ|eː|mʲ|w|r|ʏ|t|ɡ|ɣ|aː|øː|i|ɔ|p|uː|β|e|h|ŋ|q|β̞|ø|yː|a|d|ɯ|qː|ɑː)")

voiplus = FST.re("(ɟ|ɘ|y|ɢ|l|ɑ|z|n|m|j|œ|ʒ|ɫ|ɛ|o|b|oː|ɾ|u|eː|v|mʲ|w|r|ʏ|ɡ|ɣ|aː|øː|ʁ|i|ɔ|uː|β|e|ŋ|d͡ʒ|ø|yː|a|d|ɯ|ɑː)")

voiminus = FST.re("(k|c|ʂ|x|tː|f|tʲ|ʃ|t͡s|t|s|sː|t͡ʃ|p|h|t͡ɕ|q|β̞|χ|qː)")

sgminus = FST.re("(ɟ|ɘ|k|y|ɢ|l|c|ʂ|ɑ|z|n|x|m|j|œ|ʒ|ɫ|ɛ|o|tː|b|f|oː|ɾ|u|tʲ|ʃ|eː|v|mʲ|t͡s|w|r|ʏ|t|s|ɡ|ɣ|sː|aː|øː|ʁ|i|ɔ|t͡ʃ|p|uː|β|e|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|ø|yː|χ|a|d|ɯ|qː|ɑː)")

cgminus = FST.re("(ɟ|ɘ|k|y|ɢ|l|c|ʂ|ɑ|z|n|x|m|j|œ|ʒ|ɫ|ɛ|o|tː|b|f|oː|ɾ|u|tʲ|ʃ|eː|v|mʲ|t͡s|w|r|ʏ|t|s|ɡ|ɣ|sː|aː|øː|ʁ|i|ɔ|t͡ʃ|p|uː|β|e|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|ø|yː|χ|a|d|ɯ|qː|ɑː)")

antplus = FST.re("(l|z|n|m|ɫ|tː|b|f|ɾ|tʲ|v|mʲ|t͡s|r|t|s|sː|p|β|d)")

antunmarked = FST.re("(ɘ|y|ɑ|œ|ɛ|o|oː|u|eː|ʏ|aː|øː|i|ɔ|uː|e|ø|yː|a|ɯ|ɑː)")

antminus = FST.re("(ɟ|k|ɢ|c|ʂ|x|j|ʒ|ʃ|w|ɡ|ɣ|ʁ|t͡ʃ|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|χ|qː)")

corplus = FST.re("(l|ʂ|z|n|ʒ|ɫ|tː|ɾ|tʲ|ʃ|t͡s|r|t|s|sː|t͡ʃ|t͡ɕ|d͡ʒ|d)")

corminus = FST.re("(ɟ|ɘ|k|y|ɢ|c|ɑ|x|m|j|œ|ɛ|o|b|f|oː|u|eː|v|mʲ|w|ʏ|ɡ|ɣ|aː|øː|ʁ|i|ɔ|p|uː|β|e|h|ŋ|q|β̞|ø|yː|χ|a|ɯ|qː|ɑː)")

distrplus = FST.re("(ʒ|ʃ|t͡ʃ|t͡ɕ|d͡ʒ)")

distrunmarked = FST.re("(ɟ|ɘ|k|y|ɢ|c|ɑ|x|m|j|œ|ɛ|o|b|f|oː|u|eː|v|mʲ|w|ʏ|ɡ|ɣ|aː|øː|ʁ|i|ɔ|p|uː|β|e|h|ŋ|q|β̞|ø|yː|χ|a|ɯ|qː|ɑː)")

distrminus = FST.re("(l|ʂ|z|n|ɫ|tː|ɾ|tʲ|t͡s|r|t|s|sː|d)")

labplus = FST.re("(y|m|b|f|u|v|mʲ|w|p|uː|β|yː)")

labminus = FST.re("(ɟ|ɘ|k|ɢ|l|c|ʂ|ɑ|z|n|x|j|œ|ʒ|ɫ|ɛ|o|tː|oː|ɾ|tʲ|ʃ|eː|t͡s|r|ʏ|t|s|ɡ|ɣ|sː|aː|øː|ʁ|i|ɔ|t͡ʃ|e|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|ø|χ|a|d|ɯ|qː|ɑː)")

hiplus = FST.re("(ɟ|k|y|c|x|j|u|tʲ|mʲ|w|ʏ|ɡ|ɣ|i|uː|t͡ɕ|ŋ|β̞|yː|ɯ)")

hiunmarked = FST.re("(ɾ|r)")

himinus = FST.re("(ɘ|ɢ|l|ʂ|ɑ|z|n|m|œ|ʒ|ɫ|ɛ|o|tː|b|f|oː|ʃ|eː|v|t͡s|t|s|sː|aː|øː|ʁ|ɔ|t͡ʃ|p|β|e|h|q|d͡ʒ|ø|χ|a|d|qː|ɑː)")

loplus = FST.re("(ɑ|aː|a|ɑː)")

lounmarked = FST.re("(ɾ|r)")

lominus = FST.re("(ɟ|ɘ|k|y|ɢ|l|c|ʂ|z|n|x|m|j|œ|ʒ|ɫ|ɛ|o|tː|b|f|oː|u|tʲ|ʃ|eː|v|mʲ|t͡s|w|ʏ|t|s|ɡ|ɣ|sː|øː|ʁ|i|ɔ|t͡ʃ|p|uː|β|e|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|ø|yː|χ|d|ɯ|qː)")

backplus = FST.re("(k|ɢ|ɑ|x|o|oː|u|w|ɡ|ɣ|aː|ʁ|ɔ|uː|h|ŋ|q|β̞|χ|a|ɯ|qː|ɑː)")

backminus = FST.re("(ɟ|ɘ|y|l|c|ʂ|z|n|m|j|œ|ʒ|ɫ|ɛ|tː|b|f|ɾ|tʲ|ʃ|eː|v|mʲ|t͡s|r|ʏ|t|s|sː|øː|i|t͡ʃ|p|β|e|t͡ɕ|d͡ʒ|ø|yː|d)")

roundplus = FST.re("(y|œ|o|oː|u|w|ʏ|øː|ɔ|uː|ø|yː)")

roundminus = FST.re("(ɟ|ɘ|k|ɢ|l|c|ʂ|ɑ|z|n|x|m|j|ʒ|ɫ|ɛ|tː|b|f|ɾ|tʲ|ʃ|eː|v|mʲ|t͡s|r|t|s|ɡ|ɣ|sː|aː|ʁ|i|t͡ʃ|p|β|e|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|χ|a|d|ɯ|qː|ɑː)")

velaricminus = FST.re("(ɟ|ɘ|k|y|ɢ|l|c|ʂ|ɑ|z|n|x|m|j|œ|ʒ|ɫ|ɛ|o|tː|b|f|oː|ɾ|u|tʲ|ʃ|eː|v|mʲ|t͡s|w|r|ʏ|t|s|ɡ|ɣ|sː|aː|øː|ʁ|i|ɔ|t͡ʃ|p|uː|β|e|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|ø|yː|χ|a|d|ɯ|qː|ɑː)")

tenseplus = FST.re("(ɘ|y|ɑ|o|oː|u|eː|aː|øː|i|uː|e|ø|yː|a|ɑː)")

tenseunmarked = FST.re("(ɟ|k|ɢ|l|c|ʂ|z|n|x|m|j|ʒ|ɫ|tː|b|f|ɾ|tʲ|ʃ|v|mʲ|t͡s|w|r|t|s|ɡ|ɣ|sː|ʁ|t͡ʃ|p|β|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|χ|d|qː)")

tenseminus = FST.re("(œ|ɛ|ʏ|ɔ|ɯ)")

longplus = FST.re("(tː|oː|eː|sː|aː|øː|uː|yː|qː|ɑː)")

longminus = FST.re("(ɟ|ɘ|k|y|ɢ|l|c|ʂ|ɑ|z|n|x|m|j|œ|ʒ|ɫ|ɛ|o|b|f|ɾ|u|tʲ|ʃ|v|mʲ|t͡s|w|r|ʏ|t|s|ɡ|ɣ|ʁ|i|ɔ|t͡ʃ|p|β|e|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|ø|χ|a|d|ɯ)")

hitoneunmarked = FST.re("(ɟ|ɘ|k|y|ɢ|l|c|ʂ|ɑ|z|n|x|m|j|œ|ʒ|ɫ|ɛ|o|tː|b|f|oː|ɾ|u|tʲ|ʃ|eː|v|mʲ|t͡s|w|r|ʏ|t|s|ɡ|ɣ|sː|aː|øː|ʁ|i|ɔ|t͡ʃ|p|uː|β|e|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|ø|yː|χ|a|d|ɯ|qː|ɑː)")

hiregunmarked = FST.re("(ɟ|ɘ|k|y|ɢ|l|c|ʂ|ɑ|z|n|x|m|j|œ|ʒ|ɫ|ɛ|o|tː|b|f|oː|ɾ|u|tʲ|ʃ|eː|v|mʲ|t͡s|w|r|ʏ|t|s|ɡ|ɣ|sː|aː|øː|ʁ|i|ɔ|t͡ʃ|p|uː|β|e|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|ø|yː|χ|a|d|ɯ|qː|ɑː)")

trillplus = FST.re("(r)")

trillunmarked = FST.re("(ɟ|ɘ|k|y|ɢ|l|c|ʂ|ɑ|z|n|x|m|j|œ|ʒ|ɫ|ɛ|o|tː|b|f|oː|u|tʲ|ʃ|eː|v|mʲ|t͡s|w|ʏ|t|s|ɡ|ɣ|sː|aː|øː|ʁ|i|ɔ|t͡ʃ|p|uː|β|e|h|t͡ɕ|ŋ|q|β̞|d͡ʒ|ø|yː|χ|a|d|ɯ|qː|ɑː)")

trillminus = FST.re("(ɾ)")

