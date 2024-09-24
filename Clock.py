#importin whole module 
from tkinter import *
from tkinter.ttk import *
#imports strftime function to 
#retriveve system'time
from time import strftime

#create tkinker window
root = Tk()
root.title('Clock')


#this function is used to 
#display time on the label
def time():
    string = strftime('%H:%M:%S %p')
    lbl.config(text=string)
    lbl.after(1000, time)



#styling the label widget so that clock
#will look more attractive
lbl = Label(root, font=('calibri', 40, 'bold'), background='black', foreground='white')
lbl.pack(anchor='center')

# placeing clock at the centre
# of the tkinter window
time()
mainloop()