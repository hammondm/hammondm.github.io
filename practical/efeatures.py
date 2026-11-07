from pyfoma import FST

sylplus = FST.re("(ɑː|ɑ|yː|oː|øː|eː|ø|ɤː|øˑ|æː|ɑˑ|eˑ|æ|ɤˑ|yːː|iˑ|eːː|yˑ|i|uˑ|e|iː|u|æˑ|y|uː|o|ɤ|uːː|iːː|oˑ)")

sylminus = FST.re("(mˑ|pː|ŋ|tʲ|d̥ʲ|t|lː|i̯ː|sʲˑ|l|kː|tʲː|tːː|pːː|j|rː|ʃ|z̥|tˑ|u̯ː|lʲˑ|s|u̯|o̯|sˑ|ʒ̊|tː|n|k|sʲ|nʲː|hˑ|lʲː|sː|ɑ̯|m|nː|rˑ|p|v|sʲː|r|i̯|d̥|h|e̯|f|mː|lʲ|fː|o̯ː|nˑ|lˑ|b̥|ɡ̊|lʲːː|hː|ʃː|nʲ)")

sonplus = FST.re("(ɑː|mˑ|ɑ|ŋ|yː|lː|oː|øː|eː|ø|ɤː|i̯ː|øˑ|l|æː|ɑˑ|eˑ|j|rː|u̯ː|lʲˑ|æ|u̯|o̯|ɤˑ|n|yːː|nʲː|hˑ|lʲː|ɑ̯|m|iˑ|nː|eːː|rˑ|yˑ|r|i̯|h|e̯|i|uˑ|e|iː|mː|lʲ|u|æˑ|y|uː|o|ɤ|o̯ː|nˑ|lˑ|uːː|iːː|oˑ|lʲːː|hː|nʲ)")

sonminus = FST.re("(pː|tʲ|d̥ʲ|t|sʲˑ|kː|tʲː|tːː|pːː|ʃ|z̥|tˑ|s|sˑ|ʒ̊|tː|k|sʲ|sː|p|v|sʲː|d̥|f|fː|b̥|ɡ̊|ʃː)")

consplus = FST.re("(mˑ|pː|ŋ|tʲ|d̥ʲ|t|lː|sʲˑ|l|kː|tʲː|tːː|pːː|rː|ʃ|z̥|tˑ|lʲˑ|s|sˑ|ʒ̊|tː|n|k|sʲ|nʲː|hˑ|lʲː|sː|m|nː|rˑ|p|v|sʲː|r|d̥|h|f|mː|lʲ|fː|nˑ|lˑ|b̥|ɡ̊|lʲːː|hː|ʃː|nʲ)")

consminus = FST.re("(ɑː|ɑ|yː|oː|øː|eː|ø|ɤː|i̯ː|øˑ|æː|ɑˑ|eˑ|j|u̯ː|æ|u̯|o̯|ɤˑ|yːː|ɑ̯|iˑ|eːː|yˑ|i̯|e̯|i|uˑ|e|iː|u|æˑ|y|uː|o|ɤ|o̯ː|uːː|iːː|oˑ)")

contplus = FST.re("(ɑː|ɑ|yː|lː|oː|øː|eː|ø|ɤː|i̯ː|sʲˑ|øˑ|l|æː|ɑˑ|eˑ|j|rː|ʃ|z̥|u̯ː|lʲˑ|æ|s|u̯|o̯|ɤˑ|sˑ|ʒ̊|yːː|sʲ|hˑ|lʲː|sː|ɑ̯|iˑ|eːː|rˑ|yˑ|v|sʲː|r|i̯|h|e̯|i|f|uˑ|e|iː|lʲ|u|æˑ|y|uː|o|ɤ|fː|o̯ː|lˑ|uːː|iːː|oˑ|lʲːː|hː|ʃː)")

contminus = FST.re("(mˑ|pː|ŋ|tʲ|d̥ʲ|t|kː|tʲː|tːː|pːː|tˑ|tː|n|k|nʲː|m|nː|p|d̥|mː|nˑ|b̥|ɡ̊|nʲ)")

delrelunmarked = FST.re("(ɑː|ɑ|ɤː|ɑˑ|rː|ɤˑ|ɑ̯|rˑ|r|ɤ)")

delrelminus = FST.re("(mˑ|pː|ŋ|tʲ|yː|d̥ʲ|t|lː|oː|øː|eː|ø|i̯ː|sʲˑ|øˑ|l|kː|tʲː|tːː|æː|pːː|eˑ|j|ʃ|z̥|tˑ|u̯ː|lʲˑ|æ|s|u̯|o̯|sˑ|ʒ̊|tː|n|k|yːː|sʲ|nʲː|hˑ|lʲː|sː|m|iˑ|nː|eːː|yˑ|p|v|sʲː|i̯|d̥|h|e̯|i|f|uˑ|e|iː|mː|lʲ|u|æˑ|y|uː|o|fː|o̯ː|nˑ|lˑ|b̥|uːː|iːː|ɡ̊|oˑ|lʲːː|hː|ʃː|nʲ)")

latplus = FST.re("(lː|l|lʲˑ|lʲː|lʲ|lˑ|lʲːː)")

latminus = FST.re("(ɑː|mˑ|ɑ|pː|ŋ|tʲ|yː|d̥ʲ|t|oː|øː|eː|ø|ɤː|i̯ː|sʲˑ|øˑ|kː|tʲː|tːː|æː|pːː|ɑˑ|eˑ|j|rː|ʃ|z̥|tˑ|u̯ː|æ|s|u̯|o̯|ɤˑ|sˑ|ʒ̊|tː|n|k|yːː|sʲ|nʲː|hˑ|sː|ɑ̯|m|iˑ|nː|eːː|rˑ|yˑ|p|v|sʲː|r|i̯|d̥|h|e̯|i|f|uˑ|e|iː|mː|u|æˑ|y|uː|o|ɤ|fː|o̯ː|nˑ|b̥|uːː|iːː|ɡ̊|oˑ|hː|ʃː|nʲ)")

nasplus = FST.re("(mˑ|ŋ|n|nʲː|m|nː|mː|nˑ|nʲ)")

nasminus = FST.re("(ɑː|ɑ|pː|tʲ|yː|d̥ʲ|t|lː|oː|øː|eː|ø|ɤː|i̯ː|sʲˑ|øˑ|l|kː|tʲː|tːː|æː|pːː|ɑˑ|eˑ|j|rː|ʃ|z̥|tˑ|u̯ː|lʲˑ|æ|s|u̯|o̯|ɤˑ|sˑ|ʒ̊|tː|k|yːː|sʲ|hˑ|lʲː|sː|ɑ̯|iˑ|eːː|rˑ|yˑ|p|v|sʲː|r|i̯|d̥|h|e̯|i|f|uˑ|e|iː|lʲ|u|æˑ|y|uː|o|ɤ|fː|o̯ː|lˑ|b̥|uːː|iːː|ɡ̊|oˑ|lʲːː|hː|ʃː)")

stridplus = FST.re("(sʲˑ|ʃ|z̥|s|sˑ|ʒ̊|sʲ|sː|v|sʲː|f|fː|ʃː)")

stridminus = FST.re("(ɑː|mˑ|ɑ|pː|ŋ|tʲ|yː|d̥ʲ|t|lː|oː|øː|eː|ø|ɤː|i̯ː|øˑ|l|kː|tʲː|tːː|æː|pːː|ɑˑ|eˑ|j|rː|tˑ|u̯ː|lʲˑ|æ|u̯|o̯|ɤˑ|tː|n|k|yːː|nʲː|hˑ|lʲː|ɑ̯|m|iˑ|nː|eːː|rˑ|yˑ|p|r|i̯|d̥|h|e̯|i|uˑ|e|iː|mː|lʲ|u|æˑ|y|uː|o|ɤ|o̯ː|nˑ|lˑ|b̥|uːː|iːː|ɡ̊|oˑ|lʲːː|hː|nʲ)")

voiplus = FST.re("(ɑː|mˑ|ɑ|ŋ|yː|d̥ʲ|lː|oː|øː|eː|ø|ɤː|i̯ː|øˑ|l|æː|ɑˑ|eˑ|j|rː|z̥|u̯ː|lʲˑ|æ|u̯|o̯|ɤˑ|ʒ̊|n|yːː|nʲː|lʲː|ɑ̯|m|iˑ|nː|eːː|rˑ|yˑ|v|r|i̯|d̥|e̯|i|uˑ|e|iː|mː|lʲ|u|æˑ|y|uː|o|ɤ|o̯ː|nˑ|lˑ|b̥|uːː|iːː|ɡ̊|oˑ|lʲːː|nʲ)")

voiminus = FST.re("(pː|tʲ|t|sʲˑ|kː|tʲː|tːː|pːː|ʃ|tˑ|s|sˑ|tː|k|sʲ|hˑ|sː|p|sʲː|h|f|fː|hː|ʃː)")

sgminus = FST.re("(ɑː|mˑ|ɑ|pː|ŋ|tʲ|yː|d̥ʲ|t|lː|oː|øː|eː|ø|ɤː|i̯ː|sʲˑ|øˑ|l|kː|tʲː|tːː|æː|pːː|ɑˑ|eˑ|j|rː|ʃ|z̥|tˑ|u̯ː|lʲˑ|æ|s|u̯|o̯|ɤˑ|sˑ|ʒ̊|tː|n|k|yːː|sʲ|nʲː|hˑ|lʲː|sː|ɑ̯|m|iˑ|nː|eːː|rˑ|yˑ|p|v|sʲː|r|i̯|d̥|h|e̯|i|f|uˑ|e|iː|mː|lʲ|u|æˑ|y|uː|o|ɤ|fː|o̯ː|nˑ|lˑ|b̥|uːː|iːː|ɡ̊|oˑ|lʲːː|hː|ʃː|nʲ)")

cgminus = FST.re("(ɑː|mˑ|ɑ|pː|ŋ|tʲ|yː|d̥ʲ|t|lː|oː|øː|eː|ø|ɤː|i̯ː|sʲˑ|øˑ|l|kː|tʲː|tːː|æː|pːː|ɑˑ|eˑ|j|rː|ʃ|z̥|tˑ|u̯ː|lʲˑ|æ|s|u̯|o̯|ɤˑ|sˑ|ʒ̊|tː|n|k|yːː|sʲ|nʲː|hˑ|lʲː|sː|ɑ̯|m|iˑ|nː|eːː|rˑ|yˑ|p|v|sʲː|r|i̯|d̥|h|e̯|i|f|uˑ|e|iː|mː|lʲ|u|æˑ|y|uː|o|ɤ|fː|o̯ː|nˑ|lˑ|b̥|uːː|iːː|ɡ̊|oˑ|lʲːː|hː|ʃː|nʲ)")

antplus = FST.re("(mˑ|pː|tʲ|d̥ʲ|t|lː|sʲˑ|l|tʲː|tːː|pːː|rː|z̥|tˑ|lʲˑ|s|sˑ|tː|n|sʲ|nʲː|lʲː|sː|m|nː|rˑ|p|v|sʲː|r|d̥|f|mː|lʲ|fː|nˑ|lˑ|b̥|lʲːː|nʲ)")

antunmarked = FST.re("(ɑː|ɑ|yː|oː|øː|eː|ø|ɤː|i̯ː|øˑ|æː|ɑˑ|eˑ|u̯ː|æ|u̯|o̯|ɤˑ|yːː|ɑ̯|iˑ|eːː|yˑ|i̯|e̯|i|uˑ|e|iː|u|æˑ|y|uː|o|ɤ|o̯ː|uːː|iːː|oˑ)")

antminus = FST.re("(ŋ|kː|j|ʃ|ʒ̊|k|hˑ|h|ɡ̊|hː|ʃː)")

corplus = FST.re("(tʲ|d̥ʲ|t|lː|sʲˑ|l|tʲː|tːː|rː|ʃ|z̥|tˑ|lʲˑ|s|sˑ|ʒ̊|tː|n|sʲ|nʲː|lʲː|sː|nː|rˑ|sʲː|r|d̥|lʲ|nˑ|lˑ|lʲːː|ʃː|nʲ)")

corminus = FST.re("(ɑː|mˑ|ɑ|pː|ŋ|yː|oː|øː|eː|ø|ɤː|i̯ː|øˑ|kː|æː|pːː|ɑˑ|eˑ|j|u̯ː|æ|u̯|o̯|ɤˑ|k|yːː|hˑ|ɑ̯|m|iˑ|eːː|yˑ|p|v|i̯|h|e̯|i|f|uˑ|e|iː|mː|u|æˑ|y|uː|o|ɤ|fː|o̯ː|b̥|uːː|iːː|ɡ̊|oˑ|hː)")

distrplus = FST.re("(ʃ|ʒ̊|ʃː)")

distrunmarked = FST.re("(ɑː|mˑ|ɑ|pː|ŋ|yː|oː|øː|eː|ø|ɤː|i̯ː|øˑ|kː|æː|pːː|ɑˑ|eˑ|j|u̯ː|æ|u̯|o̯|ɤˑ|k|yːː|hˑ|ɑ̯|m|iˑ|eːː|yˑ|p|v|i̯|h|e̯|i|f|uˑ|e|iː|mː|u|æˑ|y|uː|o|ɤ|fː|o̯ː|b̥|uːː|iːː|ɡ̊|oˑ|hː)")

distrminus = FST.re("(tʲ|d̥ʲ|t|lː|sʲˑ|l|tʲː|tːː|rː|z̥|tˑ|lʲˑ|s|sˑ|tː|n|sʲ|nʲː|lʲː|sː|nː|rˑ|sʲː|r|d̥|lʲ|nˑ|lˑ|lʲːː|nʲ)")

labplus = FST.re("(mˑ|pː|yː|pːː|u̯ː|u̯|yːː|m|yˑ|p|v|f|uˑ|mː|u|y|uː|fː|b̥|uːː)")

labminus = FST.re("(ɑː|ɑ|ŋ|tʲ|d̥ʲ|t|lː|oː|øː|eː|ø|ɤː|i̯ː|sʲˑ|øˑ|l|kː|tʲː|tːː|æː|ɑˑ|eˑ|j|rː|ʃ|z̥|tˑ|lʲˑ|æ|s|o̯|ɤˑ|sˑ|ʒ̊|tː|n|k|sʲ|nʲː|hˑ|lʲː|sː|ɑ̯|iˑ|nː|eːː|rˑ|sʲː|r|i̯|d̥|h|e̯|i|e|iː|lʲ|æˑ|o|ɤ|o̯ː|nˑ|lˑ|iːː|ɡ̊|oˑ|lʲːː|hː|ʃː|nʲ)")

hiplus = FST.re("(ŋ|tʲ|yː|d̥ʲ|i̯ː|sʲˑ|kː|tʲː|j|u̯ː|lʲˑ|u̯|k|yːː|sʲ|nʲː|lʲː|iˑ|yˑ|sʲː|i̯|i|uˑ|iː|lʲ|u|y|uː|uːː|iːː|ɡ̊|lʲːː|nʲ)")

hiunmarked = FST.re("(rː|rˑ|r)")

himinus = FST.re("(ɑː|mˑ|ɑ|pː|t|lː|oː|øː|eː|ø|ɤː|øˑ|l|tːː|æː|pːː|ɑˑ|eˑ|ʃ|z̥|tˑ|æ|s|o̯|ɤˑ|sˑ|ʒ̊|tː|n|hˑ|sː|ɑ̯|m|nː|eːː|p|v|d̥|h|e̯|f|e|mː|æˑ|o|ɤ|fː|o̯ː|nˑ|lˑ|b̥|oˑ|hː|ʃː)")

loplus = FST.re("(ɑː|ɑ|æː|ɑˑ|æ|ɑ̯|æˑ)")

lounmarked = FST.re("(rː|rˑ|r)")

lominus = FST.re("(mˑ|pː|ŋ|tʲ|yː|d̥ʲ|t|lː|oː|øː|eː|ø|ɤː|i̯ː|sʲˑ|øˑ|l|kː|tʲː|tːː|pːː|eˑ|j|ʃ|z̥|tˑ|u̯ː|lʲˑ|s|u̯|o̯|ɤˑ|sˑ|ʒ̊|tː|n|k|yːː|sʲ|nʲː|hˑ|lʲː|sː|m|iˑ|nː|eːː|yˑ|p|v|sʲː|i̯|d̥|h|e̯|i|f|uˑ|e|iː|mː|lʲ|u|y|uː|o|ɤ|fː|o̯ː|nˑ|lˑ|b̥|uːː|iːː|ɡ̊|oˑ|lʲːː|hː|ʃː|nʲ)")

backplus = FST.re("(ɑː|ɑ|ŋ|oː|ɤː|kː|ɑˑ|u̯ː|u̯|o̯|ɤˑ|k|hˑ|ɑ̯|h|uˑ|u|uː|o|ɤ|o̯ː|uːː|ɡ̊|oˑ|hː)")

backminus = FST.re("(mˑ|pː|tʲ|yː|d̥ʲ|t|lː|øː|eː|ø|i̯ː|sʲˑ|øˑ|l|tʲː|tːː|æː|pːː|eˑ|j|rː|ʃ|z̥|tˑ|lʲˑ|æ|s|sˑ|ʒ̊|tː|n|yːː|sʲ|nʲː|lʲː|sː|m|iˑ|nː|eːː|rˑ|yˑ|p|v|sʲː|r|i̯|d̥|e̯|i|f|e|iː|mː|lʲ|æˑ|y|fː|nˑ|lˑ|b̥|iːː|lʲːː|ʃː|nʲ)")

roundplus = FST.re("(yː|oː|øː|ø|øˑ|u̯ː|u̯|o̯|yːː|yˑ|uˑ|u|y|uː|o|o̯ː|uːː|oˑ)")

roundminus = FST.re("(ɑː|mˑ|ɑ|pː|ŋ|tʲ|d̥ʲ|t|lː|eː|ɤː|i̯ː|sʲˑ|l|kː|tʲː|tːː|æː|pːː|ɑˑ|eˑ|j|rː|ʃ|z̥|tˑ|lʲˑ|æ|s|ɤˑ|sˑ|ʒ̊|tː|n|k|sʲ|nʲː|hˑ|lʲː|sː|ɑ̯|m|iˑ|nː|eːː|rˑ|p|v|sʲː|r|i̯|d̥|h|e̯|i|f|e|iː|mː|lʲ|æˑ|ɤ|fː|nˑ|lˑ|b̥|iːː|ɡ̊|lʲːː|hː|ʃː|nʲ)")

velaricminus = FST.re("(ɑː|mˑ|ɑ|pː|ŋ|tʲ|yː|d̥ʲ|t|lː|oː|øː|eː|ø|ɤː|i̯ː|sʲˑ|øˑ|l|kː|tʲː|tːː|æː|pːː|ɑˑ|eˑ|j|rː|ʃ|z̥|tˑ|u̯ː|lʲˑ|æ|s|u̯|o̯|ɤˑ|sˑ|ʒ̊|tː|n|k|yːː|sʲ|nʲː|hˑ|lʲː|sː|ɑ̯|m|iˑ|nː|eːː|rˑ|yˑ|p|v|sʲː|r|i̯|d̥|h|e̯|i|f|uˑ|e|iː|mː|lʲ|u|æˑ|y|uː|o|ɤ|fː|o̯ː|nˑ|lˑ|b̥|uːː|iːː|ɡ̊|oˑ|lʲːː|hː|ʃː|nʲ)")

tenseplus = FST.re("(ɑː|ɑ|yː|oː|øː|eː|ø|ɤː|i̯ː|øˑ|æː|ɑˑ|eˑ|u̯ː|æ|u̯|o̯|ɤˑ|yːː|ɑ̯|iˑ|eːː|yˑ|i̯|e̯|i|uˑ|e|iː|u|æˑ|y|uː|o|ɤ|o̯ː|uːː|iːː|oˑ)")

tenseunmarked = FST.re("(mˑ|pː|ŋ|tʲ|d̥ʲ|t|lː|sʲˑ|l|kː|tʲː|tːː|pːː|j|rː|ʃ|z̥|tˑ|lʲˑ|s|sˑ|ʒ̊|tː|n|k|sʲ|nʲː|hˑ|lʲː|sː|m|nː|rˑ|p|v|sʲː|r|d̥|h|f|mː|lʲ|fː|nˑ|lˑ|b̥|ɡ̊|lʲːː|hː|ʃː|nʲ)")

longplus = FST.re("(ɑː|pː|yː|lː|oː|øː|eː|ɤː|i̯ː|kː|tʲː|tːː|æː|pːː|rː|u̯ː|tː|yːː|nʲː|lʲː|sː|nː|eːː|sʲː|iː|mː|uː|fː|o̯ː|uːː|iːː|lʲːː|hː|ʃː)")

longminus = FST.re("(mˑ|ɑ|ŋ|tʲ|d̥ʲ|t|ø|sʲˑ|øˑ|l|ɑˑ|eˑ|j|ʃ|z̥|tˑ|lʲˑ|æ|s|u̯|o̯|ɤˑ|sˑ|ʒ̊|n|k|sʲ|hˑ|ɑ̯|m|iˑ|rˑ|yˑ|p|v|r|i̯|d̥|h|e̯|i|f|uˑ|e|lʲ|u|æˑ|y|o|ɤ|nˑ|lˑ|b̥|ɡ̊|oˑ|nʲ)")

hitoneunmarked = FST.re("(ɑː|mˑ|ɑ|pː|ŋ|tʲ|yː|d̥ʲ|t|lː|oː|øː|eː|ø|ɤː|i̯ː|sʲˑ|øˑ|l|kː|tʲː|tːː|æː|pːː|ɑˑ|eˑ|j|rː|ʃ|z̥|tˑ|u̯ː|lʲˑ|æ|s|u̯|o̯|ɤˑ|sˑ|ʒ̊|tː|n|k|yːː|sʲ|nʲː|hˑ|lʲː|sː|ɑ̯|m|iˑ|nː|eːː|rˑ|yˑ|p|v|sʲː|r|i̯|d̥|h|e̯|i|f|uˑ|e|iː|mː|lʲ|u|æˑ|y|uː|o|ɤ|fː|o̯ː|nˑ|lˑ|b̥|uːː|iːː|ɡ̊|oˑ|lʲːː|hː|ʃː|nʲ)")

hiregunmarked = FST.re("(ɑː|mˑ|ɑ|pː|ŋ|tʲ|yː|d̥ʲ|t|lː|oː|øː|eː|ø|ɤː|i̯ː|sʲˑ|øˑ|l|kː|tʲː|tːː|æː|pːː|ɑˑ|eˑ|j|rː|ʃ|z̥|tˑ|u̯ː|lʲˑ|æ|s|u̯|o̯|ɤˑ|sˑ|ʒ̊|tː|n|k|yːː|sʲ|nʲː|hˑ|lʲː|sː|ɑ̯|m|iˑ|nː|eːː|rˑ|yˑ|p|v|sʲː|r|i̯|d̥|h|e̯|i|f|uˑ|e|iː|mː|lʲ|u|æˑ|y|uː|o|ɤ|fː|o̯ː|nˑ|lˑ|b̥|uːː|iːː|ɡ̊|oˑ|lʲːː|hː|ʃː|nʲ)")

halflongplus = FST.re("(mˑ|sʲˑ|øˑ|tːː|pːː|ɑˑ|eˑ|tˑ|lʲˑ|ɤˑ|sˑ|yːː|hˑ|iˑ|eːː|rˑ|yˑ|uˑ|æˑ|nˑ|lˑ|uːː|iːː|oˑ|lʲːː)")

halflongminus = FST.re("(ɑː|ɑ|pː|ŋ|tʲ|yː|d̥ʲ|t|lː|oː|øː|eː|ø|ɤː|i̯ː|l|kː|tʲː|æː|j|rː|ʃ|z̥|u̯ː|æ|s|u̯|o̯|ʒ̊|tː|n|k|sʲ|nʲː|lʲː|sː|ɑ̯|m|nː|p|v|sʲː|r|i̯|d̥|h|e̯|i|f|e|iː|mː|lʲ|u|y|uː|o|ɤ|fː|o̯ː|b̥|ɡ̊|hː|ʃː|nʲ)")

