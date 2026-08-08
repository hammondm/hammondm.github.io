from tkinter import *
from tkinter import filedialog
from playsound import playsound
import sys,re

#specify the font for everything
myfont = ('Helvetica',20)

#globals
wavfile = ''
results = []

#search function
def search(pat,loc):
	filename = loc + 'validated.tsv'
	f = open(filename,'r')
	t = f.read()
	f.close()
	t = t.split('\n')
	t = t[1:-1]
	results = []
	for line in t:
		bits = line.split('\t')
		#may need to change these two for different languages
		wavfile = bits[1]
		sentence = bits[3]
		if re.search(pat,sentence):
			results.append((wavfile,sentence))
	return results

#function to specify location of files and metadata
def getdir():
	res = filedialog.askdirectory()
	if len(res) > 0:
		filedir.set(res)
		#enable searches now
		searchfield.config(state='normal')

#start GUI
r = Tk()
r.title('Listen to Common Voice files')

#store search term
searchVar = StringVar()

#default text for label for file location
filedir = StringVar()
filedir.set('File location not yet specified')

#make a window
f = Frame(r)
f.pack()

#label for location of common voice files
cvlab = Label(
	f,
	font=myfont,
	text='Click button to specify location of Common Voice data'
)
cvlab.pack()

#buton for specifying file location
filebutton = Button(
	f,
	font=myfont,
	text='Files',
	command=getdir
)
filebutton.pack()

#current file location
loclab = Label(
	f,
	font=myfont,
	textvariable=filedir
)
loclab.pack()

#label for search
relab = Label(
	f,
	font=myfont,
	text='Enter a search and hit return'
)
relab.pack()

#textfield for search
searchfield = Entry(
	f,
	font=myfont,
	textvariable=searchVar,
	state='disabled',
	width=20
)
searchfield.pack()

#can specify file location on the command-line
if len(sys.argv) > 1:
	filedirstr = sys.argv[1]
	searchfield.config(state='normal')
	filedir.set(filedirstr)

#do search on return
def searchstr(_):
	global results
	pat = searchVar.get()
	dirname = filedir.get()
	if dirname[-1] != '/': dirname += '/'
	res = search(pat,dirname)
	results = res
	filebox.delete(0,END)
	for result in results:
		filebox.insert(END, result[1])

#search when user types return
searchfield.bind('<Return>',searchstr)

#instructions for listening to an item
listenlab = Label(
	f,
	text='Double-click on an item to hear it',
	font=myfont
)
listenlab.pack()

#scrollable list of wavefiles in separate subframe
fsub = Frame(f)
fsub.pack()
#the clickable list
filebox = Listbox(
	fsub,
	width=70,
	font=myfont,
	height=10
)
filebox.pack(side='left',fill='y')
#scrollbar for the list of files
scrollbar = Scrollbar(
	fsub,
	orient="vertical",
	command=filebox.yview
)
scrollbar.pack(side="right",fill="y")
filebox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=filebox.yview)

#quit button
quitbutton = Button(
	f,
	text='Quit',
	font=myfont,
	command=quit
)
quitbutton.pack(side='right')

#play an item
def getitem(x):
	global wavfile
	res = filebox.curselection()[0]
	wavfile = results[res][0]
	playsound(filedir.get()+'/clips/'+wavfile)

#double-click on an item to play it
filebox.bind('<Double-Button>',getitem)

#wait for things to happen
mainloop()

