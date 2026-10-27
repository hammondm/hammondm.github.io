from pyfoma import FST

sylplus = FST.re("(øˑ|eˑ|ɤ|uˑ|ɑ|iˑ|y|oː|uːː|e|o|eːː|ø|æː|uː|ɤˑ|ɤː|yː|yːː|ɑˑ|æ|iː|yˑ|øː|oˑ|i|eː|iːː|u|æˑ|ɑː)")

sylminus = FST.re("(ʃː|nʲ|t|sː|b̥|k|z̥|nː|lʲˑ|lː|lˑ|ɑ̯|tʲː|tːː|h|d̥|mː|ʃ|lʲː|kː|nʲː|i̯|d̥ʲ|rː|ʒ̊|tʲ|m|u̯|o̯|lʲ|e̯|n|tˑ|u̯ː|v|rˑ|sˑ|pːː|hˑ|s|l|i̯ː|nˑ|tː|ɡ̊|fː|ŋ|sʲ|j|hː|pː|lʲːː|o̯ː|mˑ|sʲˑ|p|f|sʲː|r)")

sonplus = FST.re("(øˑ|nʲ|eˑ|ɤ|uˑ|ɑ|nː|iˑ|lʲˑ|lː|lˑ|ɑ̯|y|h|mː|lʲː|oː|nʲː|i̯|rː|uːː|e|m|u̯|o|eːː|o̯|lʲ|ø|e̯|n|æː|uː|u̯ː|rˑ|ɤˑ|hˑ|ɤː|l|i̯ː|nˑ|yː|yːː|ɑˑ|æ|ŋ|j|iː|hː|lʲːː|o̯ː|yˑ|øː|oˑ|i|mˑ|eː|iːː|u|æˑ|ɑː|r)")

sonminus = FST.re("(ʃː|t|sː|b̥|k|z̥|tʲː|tːː|d̥|ʃ|kː|d̥ʲ|ʒ̊|tʲ|tˑ|v|sˑ|pːː|s|tː|ɡ̊|fː|sʲ|pː|sʲˑ|p|f|sʲː)")

consplus = FST.re("(ʃː|nʲ|t|sː|b̥|k|z̥|nː|lʲˑ|lː|lˑ|tʲː|tːː|h|d̥|mː|ʃ|lʲː|kː|nʲː|d̥ʲ|rː|ʒ̊|tʲ|m|lʲ|n|tˑ|v|rˑ|sˑ|pːː|hˑ|s|l|nˑ|tː|ɡ̊|fː|ŋ|sʲ|hː|pː|lʲːː|mˑ|sʲˑ|p|f|sʲː|r)")

consminus = FST.re("(øˑ|eˑ|ɤ|uˑ|ɑ|iˑ|ɑ̯|y|oː|i̯|uːː|e|u̯|o|eːː|o̯|ø|e̯|æː|uː|u̯ː|ɤˑ|ɤː|i̯ː|yː|yːː|ɑˑ|æ|j|iː|o̯ː|yˑ|øː|oˑ|i|eː|iːː|u|æˑ|ɑː)")

contplus = FST.re("(øˑ|ʃː|eˑ|sː|ɤ|uˑ|ɑ|z̥|iˑ|lʲˑ|lː|lˑ|ɑ̯|y|h|ʃ|lʲː|oː|i̯|rː|uːː|ʒ̊|e|u̯|o|eːː|o̯|lʲ|ø|e̯|æː|uː|u̯ː|v|rˑ|sˑ|ɤˑ|hˑ|ɤː|s|l|i̯ː|yː|fː|yːː|ɑˑ|æ|sʲ|j|iː|hː|lʲːː|o̯ː|yˑ|øː|oˑ|i|eː|sʲˑ|iːː|u|æˑ|f|ɑː|sʲː|r)")

contminus = FST.re("(nʲ|t|b̥|k|nː|tʲː|tːː|d̥|mː|kː|nʲː|d̥ʲ|tʲ|m|n|tˑ|pːː|nˑ|tː|ɡ̊|ŋ|pː|mˑ|p)")

delrelunmarked = FST.re("(ɤ|ɑ|ɑ̯|rː|rˑ|ɤˑ|ɤː|ɑˑ|ɑː|r)")

delrelminus = FST.re("(øˑ|ʃː|nʲ|t|eˑ|sː|b̥|uˑ|k|z̥|nː|iˑ|lʲˑ|lː|lˑ|y|tʲː|tːː|h|d̥|mː|ʃ|lʲː|oː|kː|nʲː|i̯|d̥ʲ|uːː|ʒ̊|tʲ|e|m|u̯|o|eːː|o̯|lʲ|ø|e̯|n|æː|uː|tˑ|u̯ː|v|sˑ|pːː|hˑ|s|l|i̯ː|nˑ|tː|yː|ɡ̊|fː|yːː|æ|ŋ|sʲ|j|iː|hː|pː|lʲːː|o̯ː|yˑ|øː|oˑ|i|mˑ|eː|sʲˑ|iːː|u|æˑ|p|f|sʲː)")

latplus = FST.re("(lʲˑ|lː|lˑ|lʲː|lʲ|l|lʲːː)")

latminus = FST.re("(øˑ|ʃː|nʲ|t|eˑ|sː|b̥|ɤ|uˑ|ɑ|k|z̥|nː|iˑ|ɑ̯|y|tʲː|tːː|h|d̥|mː|ʃ|oː|kː|nʲː|i̯|d̥ʲ|rː|uːː|ʒ̊|tʲ|e|m|u̯|o|eːː|o̯|ø|e̯|n|æː|uː|tˑ|u̯ː|v|rˑ|sˑ|ɤˑ|pːː|hˑ|ɤː|s|i̯ː|nˑ|tː|yː|ɡ̊|fː|yːː|ɑˑ|æ|ŋ|sʲ|j|iː|hː|pː|o̯ː|yˑ|øː|oˑ|i|mˑ|eː|sʲˑ|iːː|u|æˑ|p|f|ɑː|sʲː|r)")

nasplus = FST.re("(nʲ|nː|mː|nʲː|m|n|nˑ|ŋ|mˑ)")

nasminus = FST.re("(øˑ|ʃː|t|eˑ|sː|b̥|ɤ|uˑ|ɑ|k|z̥|iˑ|lʲˑ|lː|lˑ|ɑ̯|y|tʲː|tːː|h|d̥|ʃ|lʲː|oː|kː|i̯|d̥ʲ|rː|uːː|ʒ̊|tʲ|e|u̯|o|eːː|o̯|lʲ|ø|e̯|æː|uː|tˑ|u̯ː|v|rˑ|sˑ|ɤˑ|pːː|hˑ|ɤː|s|l|i̯ː|tː|yː|ɡ̊|fː|yːː|ɑˑ|æ|sʲ|j|iː|hː|pː|lʲːː|o̯ː|yˑ|øː|oˑ|i|eː|sʲˑ|iːː|u|æˑ|p|f|ɑː|sʲː|r)")

stridplus = FST.re("(ʃː|sː|z̥|ʃ|ʒ̊|v|sˑ|s|fː|sʲ|sʲˑ|f|sʲː)")

stridminus = FST.re("(øˑ|nʲ|t|eˑ|b̥|ɤ|uˑ|ɑ|k|nː|iˑ|lʲˑ|lː|lˑ|ɑ̯|y|tʲː|tːː|h|d̥|mː|lʲː|oː|kː|nʲː|i̯|d̥ʲ|rː|uːː|tʲ|e|m|u̯|o|eːː|o̯|lʲ|ø|e̯|n|æː|uː|tˑ|u̯ː|rˑ|ɤˑ|pːː|hˑ|ɤː|l|i̯ː|nˑ|tː|yː|ɡ̊|yːː|ɑˑ|æ|ŋ|j|iː|hː|pː|lʲːː|o̯ː|yˑ|øː|oˑ|i|mˑ|eː|iːː|u|æˑ|p|ɑː|r)")

voiplus = FST.re("(øˑ|nʲ|eˑ|b̥|ɤ|uˑ|ɑ|z̥|nː|iˑ|lʲˑ|lː|lˑ|ɑ̯|y|d̥|mː|lʲː|oː|nʲː|i̯|d̥ʲ|rː|uːː|ʒ̊|e|m|u̯|o|eːː|o̯|lʲ|ø|e̯|n|æː|uː|u̯ː|v|rˑ|ɤˑ|ɤː|l|i̯ː|nˑ|yː|ɡ̊|yːː|ɑˑ|æ|ŋ|j|iː|lʲːː|o̯ː|yˑ|øː|oˑ|i|mˑ|eː|iːː|u|æˑ|ɑː|r)")

voiminus = FST.re("(ʃː|t|sː|k|tʲː|tːː|h|ʃ|kː|tʲ|tˑ|sˑ|pːː|hˑ|s|tː|fː|sʲ|hː|pː|sʲˑ|p|f|sʲː)")

sgminus = FST.re("(øˑ|ʃː|nʲ|t|eˑ|sː|b̥|ɤ|uˑ|ɑ|k|z̥|nː|iˑ|lʲˑ|lː|lˑ|ɑ̯|y|tʲː|tːː|h|d̥|mː|ʃ|lʲː|oː|kː|nʲː|i̯|d̥ʲ|rː|uːː|ʒ̊|tʲ|e|m|u̯|o|eːː|o̯|lʲ|ø|e̯|n|æː|uː|tˑ|u̯ː|v|rˑ|sˑ|ɤˑ|pːː|hˑ|ɤː|s|l|i̯ː|nˑ|tː|yː|ɡ̊|fː|yːː|ɑˑ|æ|ŋ|sʲ|j|iː|hː|pː|lʲːː|o̯ː|yˑ|øː|oˑ|i|mˑ|eː|sʲˑ|iːː|u|æˑ|p|f|ɑː|sʲː|r)")

cgminus = FST.re("(øˑ|ʃː|nʲ|t|eˑ|sː|b̥|ɤ|uˑ|ɑ|k|z̥|nː|iˑ|lʲˑ|lː|lˑ|ɑ̯|y|tʲː|tːː|h|d̥|mː|ʃ|lʲː|oː|kː|nʲː|i̯|d̥ʲ|rː|uːː|ʒ̊|tʲ|e|m|u̯|o|eːː|o̯|lʲ|ø|e̯|n|æː|uː|tˑ|u̯ː|v|rˑ|sˑ|ɤˑ|pːː|hˑ|ɤː|s|l|i̯ː|nˑ|tː|yː|ɡ̊|fː|yːː|ɑˑ|æ|ŋ|sʲ|j|iː|hː|pː|lʲːː|o̯ː|yˑ|øː|oˑ|i|mˑ|eː|sʲˑ|iːː|u|æˑ|p|f|ɑː|sʲː|r)")

antplus = FST.re("(nʲ|t|sː|b̥|z̥|nː|lʲˑ|lː|lˑ|tʲː|tːː|d̥|mː|lʲː|nʲː|d̥ʲ|rː|tʲ|m|lʲ|n|tˑ|v|rˑ|sˑ|pːː|s|l|nˑ|tː|fː|sʲ|pː|lʲːː|mˑ|sʲˑ|p|f|sʲː|r)")

antunmarked = FST.re("(øˑ|eˑ|ɤ|uˑ|ɑ|iˑ|ɑ̯|y|oː|i̯|uːː|e|u̯|o|eːː|o̯|ø|e̯|æː|uː|u̯ː|ɤˑ|ɤː|i̯ː|yː|yːː|ɑˑ|æ|iː|o̯ː|yˑ|øː|oˑ|i|eː|iːː|u|æˑ|ɑː)")

antminus = FST.re("(ʃː|k|h|ʃ|kː|ʒ̊|hˑ|ɡ̊|ŋ|j|hː)")

corplus = FST.re("(ʃː|nʲ|t|sː|z̥|nː|lʲˑ|lː|lˑ|tʲː|tːː|d̥|ʃ|lʲː|nʲː|d̥ʲ|rː|ʒ̊|tʲ|lʲ|n|tˑ|rˑ|sˑ|s|l|nˑ|tː|sʲ|lʲːː|sʲˑ|sʲː|r)")

corminus = FST.re("(øˑ|eˑ|b̥|ɤ|uˑ|ɑ|k|iˑ|ɑ̯|y|h|mː|oː|kː|i̯|uːː|e|m|u̯|o|eːː|o̯|ø|e̯|æː|uː|u̯ː|v|ɤˑ|pːː|hˑ|ɤː|i̯ː|yː|ɡ̊|fː|yːː|ɑˑ|æ|ŋ|j|iː|hː|pː|o̯ː|yˑ|øː|oˑ|i|mˑ|eː|iːː|u|æˑ|p|f|ɑː)")

distrplus = FST.re("(ʃː|ʃ|ʒ̊)")

distrunmarked = FST.re("(øˑ|eˑ|b̥|ɤ|uˑ|ɑ|k|iˑ|ɑ̯|y|h|mː|oː|kː|i̯|uːː|e|m|u̯|o|eːː|o̯|ø|e̯|æː|uː|u̯ː|v|ɤˑ|pːː|hˑ|ɤː|i̯ː|yː|ɡ̊|fː|yːː|ɑˑ|æ|ŋ|j|iː|hː|pː|o̯ː|yˑ|øː|oˑ|i|mˑ|eː|iːː|u|æˑ|p|f|ɑː)")

distrminus = FST.re("(nʲ|t|sː|z̥|nː|lʲˑ|lː|lˑ|tʲː|tːː|d̥|lʲː|nʲː|d̥ʲ|rː|tʲ|lʲ|n|tˑ|rˑ|sˑ|s|l|nˑ|tː|sʲ|lʲːː|sʲˑ|sʲː|r)")

labplus = FST.re("(b̥|uˑ|y|mː|uːː|m|u̯|uː|u̯ː|v|pːː|yː|fː|yːː|pː|yˑ|mˑ|u|p|f)")

labminus = FST.re("(øˑ|ʃː|nʲ|t|eˑ|sː|ɤ|ɑ|k|z̥|nː|iˑ|lʲˑ|lː|lˑ|ɑ̯|tʲː|tːː|h|d̥|ʃ|lʲː|oː|kː|nʲː|i̯|d̥ʲ|rː|ʒ̊|tʲ|e|o|eːː|o̯|lʲ|ø|e̯|n|æː|tˑ|rˑ|sˑ|ɤˑ|hˑ|ɤː|s|l|i̯ː|nˑ|tː|ɡ̊|ɑˑ|æ|ŋ|sʲ|j|iː|hː|lʲːː|o̯ː|øː|oˑ|i|eː|sʲˑ|iːː|æˑ|ɑː|sʲː|r)")

hiplus = FST.re("(nʲ|uˑ|k|iˑ|lʲˑ|y|tʲː|lʲː|kː|nʲː|i̯|d̥ʲ|uːː|tʲ|u̯|lʲ|uː|u̯ː|i̯ː|yː|ɡ̊|yːː|ŋ|sʲ|j|iː|lʲːː|yˑ|i|sʲˑ|iːː|u|sʲː)")

hiunmarked = FST.re("(rː|rˑ|r)")

himinus = FST.re("(øˑ|ʃː|t|eˑ|sː|b̥|ɤ|ɑ|z̥|nː|lː|lˑ|ɑ̯|tːː|h|d̥|mː|ʃ|oː|ʒ̊|e|m|o|eːː|o̯|ø|e̯|n|æː|tˑ|v|sˑ|ɤˑ|pːː|hˑ|ɤː|s|l|nˑ|tː|fː|ɑˑ|æ|hː|pː|o̯ː|øː|oˑ|mˑ|eː|æˑ|p|f|ɑː)")

loplus = FST.re("(ɑ|ɑ̯|æː|ɑˑ|æ|æˑ|ɑː)")

lounmarked = FST.re("(rː|rˑ|r)")

lominus = FST.re("(øˑ|ʃː|nʲ|t|eˑ|sː|b̥|ɤ|uˑ|k|z̥|nː|iˑ|lʲˑ|lː|lˑ|y|tʲː|tːː|h|d̥|mː|ʃ|lʲː|oː|kː|nʲː|i̯|d̥ʲ|uːː|ʒ̊|tʲ|e|m|u̯|o|eːː|o̯|lʲ|ø|e̯|n|uː|tˑ|u̯ː|v|sˑ|ɤˑ|pːː|hˑ|ɤː|s|l|i̯ː|nˑ|tː|yː|ɡ̊|fː|yːː|ŋ|sʲ|j|iː|hː|pː|lʲːː|o̯ː|yˑ|øː|oˑ|i|mˑ|eː|sʲˑ|iːː|u|p|f|sʲː)")

backplus = FST.re("(ɤ|uˑ|ɑ|k|ɑ̯|h|oː|kː|uːː|u̯|o|o̯|uː|u̯ː|ɤˑ|hˑ|ɤː|ɡ̊|ɑˑ|ŋ|hː|o̯ː|oˑ|u|ɑː)")

backminus = FST.re("(øˑ|ʃː|nʲ|t|eˑ|sː|b̥|z̥|nː|iˑ|lʲˑ|lː|lˑ|y|tʲː|tːː|d̥|mː|ʃ|lʲː|nʲː|i̯|d̥ʲ|rː|ʒ̊|tʲ|e|m|eːː|lʲ|ø|e̯|n|æː|tˑ|v|rˑ|sˑ|pːː|s|l|i̯ː|nˑ|tː|yː|fː|yːː|æ|sʲ|j|iː|pː|lʲːː|yˑ|øː|i|mˑ|eː|sʲˑ|iːː|æˑ|p|f|sʲː|r)")

roundplus = FST.re("(øˑ|uˑ|y|oː|uːː|u̯|o|o̯|ø|uː|u̯ː|yː|yːː|o̯ː|yˑ|øː|oˑ|u)")

roundminus = FST.re("(ʃː|nʲ|t|eˑ|sː|b̥|ɤ|ɑ|k|z̥|nː|iˑ|lʲˑ|lː|lˑ|ɑ̯|tʲː|tːː|h|d̥|mː|ʃ|lʲː|kː|nʲː|i̯|d̥ʲ|rː|ʒ̊|tʲ|e|m|eːː|lʲ|e̯|n|æː|tˑ|v|rˑ|sˑ|ɤˑ|pːː|hˑ|ɤː|s|l|i̯ː|nˑ|tː|ɡ̊|fː|ɑˑ|æ|ŋ|sʲ|j|iː|hː|pː|lʲːː|i|mˑ|eː|sʲˑ|iːː|æˑ|p|f|ɑː|sʲː|r)")

velaricminus = FST.re("(øˑ|ʃː|nʲ|t|eˑ|sː|b̥|ɤ|uˑ|ɑ|k|z̥|nː|iˑ|lʲˑ|lː|lˑ|ɑ̯|y|tʲː|tːː|h|d̥|mː|ʃ|lʲː|oː|kː|nʲː|i̯|d̥ʲ|rː|uːː|ʒ̊|tʲ|e|m|u̯|o|eːː|o̯|lʲ|ø|e̯|n|æː|uː|tˑ|u̯ː|v|rˑ|sˑ|ɤˑ|pːː|hˑ|ɤː|s|l|i̯ː|nˑ|tː|yː|ɡ̊|fː|yːː|ɑˑ|æ|ŋ|sʲ|j|iː|hː|pː|lʲːː|o̯ː|yˑ|øː|oˑ|i|mˑ|eː|sʲˑ|iːː|u|æˑ|p|f|ɑː|sʲː|r)")

tenseplus = FST.re("(øˑ|eˑ|ɤ|uˑ|ɑ|iˑ|ɑ̯|y|oː|i̯|uːː|e|u̯|o|eːː|o̯|ø|e̯|æː|uː|u̯ː|ɤˑ|ɤː|i̯ː|yː|yːː|ɑˑ|æ|iː|o̯ː|yˑ|øː|oˑ|i|eː|iːː|u|æˑ|ɑː)")

tenseunmarked = FST.re("(ʃː|nʲ|t|sː|b̥|k|z̥|nː|lʲˑ|lː|lˑ|tʲː|tːː|h|d̥|mː|ʃ|lʲː|kː|nʲː|d̥ʲ|rː|ʒ̊|tʲ|m|lʲ|n|tˑ|v|rˑ|sˑ|pːː|hˑ|s|l|nˑ|tː|ɡ̊|fː|ŋ|sʲ|j|hː|pː|lʲːː|mˑ|sʲˑ|p|f|sʲː|r)")

longplus = FST.re("(ʃː|sː|nː|lː|tʲː|tːː|mː|lʲː|oː|kː|nʲː|rː|uːː|eːː|æː|uː|u̯ː|pːː|ɤː|i̯ː|tː|yː|fː|yːː|iː|hː|pː|lʲːː|o̯ː|øː|eː|iːː|ɑː|sʲː)")

longminus = FST.re("(øˑ|nʲ|t|eˑ|b̥|ɤ|uˑ|ɑ|k|z̥|iˑ|lʲˑ|lˑ|ɑ̯|y|h|d̥|ʃ|i̯|d̥ʲ|ʒ̊|tʲ|e|m|u̯|o|o̯|lʲ|ø|e̯|n|tˑ|v|rˑ|sˑ|ɤˑ|hˑ|s|l|nˑ|ɡ̊|ɑˑ|æ|ŋ|sʲ|j|yˑ|oˑ|i|mˑ|sʲˑ|u|æˑ|p|f|r)")

hitoneunmarked = FST.re("(øˑ|ʃː|nʲ|t|eˑ|sː|b̥|ɤ|uˑ|ɑ|k|z̥|nː|iˑ|lʲˑ|lː|lˑ|ɑ̯|y|tʲː|tːː|h|d̥|mː|ʃ|lʲː|oː|kː|nʲː|i̯|d̥ʲ|rː|uːː|ʒ̊|tʲ|e|m|u̯|o|eːː|o̯|lʲ|ø|e̯|n|æː|uː|tˑ|u̯ː|v|rˑ|sˑ|ɤˑ|pːː|hˑ|ɤː|s|l|i̯ː|nˑ|tː|yː|ɡ̊|fː|yːː|ɑˑ|æ|ŋ|sʲ|j|iː|hː|pː|lʲːː|o̯ː|yˑ|øː|oˑ|i|mˑ|eː|sʲˑ|iːː|u|æˑ|p|f|ɑː|sʲː|r)")

hiregunmarked = FST.re("(øˑ|ʃː|nʲ|t|eˑ|sː|b̥|ɤ|uˑ|ɑ|k|z̥|nː|iˑ|lʲˑ|lː|lˑ|ɑ̯|y|tʲː|tːː|h|d̥|mː|ʃ|lʲː|oː|kː|nʲː|i̯|d̥ʲ|rː|uːː|ʒ̊|tʲ|e|m|u̯|o|eːː|o̯|lʲ|ø|e̯|n|æː|uː|tˑ|u̯ː|v|rˑ|sˑ|ɤˑ|pːː|hˑ|ɤː|s|l|i̯ː|nˑ|tː|yː|ɡ̊|fː|yːː|ɑˑ|æ|ŋ|sʲ|j|iː|hː|pː|lʲːː|o̯ː|yˑ|øː|oˑ|i|mˑ|eː|sʲˑ|iːː|u|æˑ|p|f|ɑː|sʲː|r)")

halflongplus = FST.re("(øˑ|eˑ|uˑ|iˑ|lʲˑ|lˑ|tːː|uːː|eːː|tˑ|rˑ|sˑ|ɤˑ|pːː|hˑ|nˑ|yːː|ɑˑ|lʲːː|yˑ|oˑ|mˑ|sʲˑ|iːː|æˑ)")

halflongminus = FST.re("(ʃː|nʲ|t|sː|b̥|ɤ|ɑ|k|z̥|nː|lː|ɑ̯|y|tʲː|h|d̥|mː|ʃ|lʲː|oː|kː|nʲː|i̯|d̥ʲ|rː|ʒ̊|tʲ|e|m|u̯|o|o̯|lʲ|ø|e̯|n|æː|uː|u̯ː|v|ɤː|s|l|i̯ː|tː|yː|ɡ̊|fː|æ|ŋ|sʲ|j|iː|hː|pː|o̯ː|øː|i|eː|u|p|f|ɑː|sʲː|r)")

