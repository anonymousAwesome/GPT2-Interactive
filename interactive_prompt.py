# Requires the huggingface library and either Tensorflow or Pytorch

'''
[l]oop and [r]and only work if the percentage is some absurdly large
number like 1000.  Otherwise it usually picks the most likely of the low-
probability results.
'''

# newline token is number 198
# double-newline is 628

import torch
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel

'''
Note: because my internet connection was unreliable I downloaded the file
manually and saved it to a folder listed below.  If you need to do the 
same, comment out the first pair of lines, uncomment the second pair,
and save the files to a folder named distilgpt2 that exists in the same
folder as this .py file.
'''
 
#automatic download
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

# pre-downloaded file
# tokenizer = GPT2Tokenizer.from_pretrained('./distilgpt2/')
# model = GPT2LMHeadModel.from_pretrained('./distilgpt2/')


# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

percent_limit=.8

select_range=10 #from top to select_range, and from bottom to (bottom minus select_range)

def append_newlines(file_contents):
	untruncated_tokens=tokenizer.encode(file_contents)
	extra_newlines=len(file_contents)-len(tokenizer.decode(untruncated_tokens))
	untruncated_tokens.extend([198]*(extra_newlines))
	return untruncated_tokens

def check_percent(data_tensor,limit):
	for i in range(len(data_tensor)):
		if data_tensor[i][1][0].item()>=limit:
			return i

def rand_gen(data_tensor,top_p):
	number=random.random()
	number=number*top_p
	return check_percent(data_tensor, number)	

def generate_data():
	with open ("prompt.txt") as f:
		file_contents=f.read()
		
	untruncated_tokens=append_newlines(file_contents)
	truncated_tokens=untruncated_tokens[-1022:] # <--largest possible quantity of tokens, minus one or two for safe margin.
	#truncated_tokens=untruncated_tokens[-500:]
	print(tokenizer.decode(truncated_tokens)+"█")
	tokens_tensor = torch.tensor([truncated_tokens])
	# Predict all tokens
	with torch.no_grad():
		outputs = model(tokens_tensor)
		predictions = outputs[0]
	
	#>>> predictions.size()
	#torch.Size([1, 500, 50257])
	#
	# 1 and 500 correspond to the input tensor.  The 500 comes from the fact
	# that I'm inputting the maximum 500 tokens.  50257 is the vocabulary
	# size of GPT-2.
	combined=torch.sort(predictions.squeeze()[-1])
	# -1 refers to the last dimension (499 in this case), -2 is the one
	# before that, etc.
	# converting the logits to percentage with the softmax function
	percent=torch.nn.functional.softmax(combined[0],dim=0)
	
	# combining percent with the rest of the data
	combined2=torch.stack((percent,combined[1].float()))

	#flipping the dimensions to make it easier to work with
	combined3=torch.transpose(combined2,0,1)
	combined4=torch.flip(combined3,[0,1])
	
	cumulative=combined4[:,1].cumsum(0)
	
	cumulative2=cumulative.unsqueeze(1)
	
	combined5=torch.cat((cumulative2,combined4),dim=1)

	return list(enumerate(combined5))


def post_data(prompt):
	with open ("prompt.txt") as f:
		file_contents=f.read()
		
	file_contents+=prompt
	with open ("prompt.txt","w") as f:
		f.write(file_contents)


while True:

	enumerated=generate_data()
	
	results=input("What percent should I show you? (current: "+str(percent_limit)+") ")
	try:
		if results!="":
			percent_limit=float(results)
	except ValueError:
		pass	
	except TypeError:
		pass

	results=input("What range should I display from the top and bottom? (current: "+str(select_range)+") ")
	try:
		if results!="":
			select_range=int(results)
	except ValueError:
		pass	
	except TypeError:
		pass
		
	top_n_size=check_percent(enumerated,percent_limit)+1

	enumerated=enumerated[0:top_n_size]
	
	enum_top=enumerated[0:select_range]
	enum_bottom=enumerated[top_n_size-select_range:top_n_size]
	
	enum_combined=enum_top+enum_bottom
	
	unique_list=[]
	for i in enum_combined:
		if i not in unique_list:
			unique_list.append(i)
	
	enumerated=unique_list
	
	for i in range(len(enumerated)):
		print(i,enumerated[i][0], round(enumerated[i][1][2].item(),4), round(enumerated[i][1][0].item(),4), tokenizer.decode(int(enumerated[i][1][1].item())))

#	for i in range(top_n_size):
#		if random.random()<=enumerated[i][1][1].item()*10:
#			print(enumerated[i][0], round(enumerated[i][1][1].item(),2), tokenizer.decode(int(enumerated[i][1][0].item())))

	prompt=input("Options: Select a number, enter custom text, [r]and, or [l]oop ")
	try:
		prompt=int(prompt)
	except ValueError:
		pass

	if (isinstance(prompt,int)):
		prompt=tokenizer.decode(int(enumerated[prompt][1][1].item()))
		post_data(prompt)
	
	elif prompt=="r":
		again=True
		while again==True:
			random_index=rand_gen(enumerated,percent_limit)
			random_word=tokenizer.decode(int(enumerated[random_index][1][1].item()))
			print(enumerated[random_index][0], round(enumerated[random_index][1][2].item(),4), round(enumerated[random_index][1][0].item(),4), random_word)
			prompt2=input("Use this? [y/N], #")
			try:
				prompt2=int(prompt2)
				prompt=tokenizer.decode(int(enumerated[prompt2][1][1].item()))
				again=False
			except ValueError:
				pass
			except TypeError:
				pass
			if prompt2=="y":
				prompt=random_word
				again=False
			elif again==True:
				prompt3=input("Another random word? [Y/n], #")
				try:
					prompt3=int(prompt3)
					prompt=tokenizer.decode(int(enumerated[prompt3][1][1].item()))
					again=False
				except ValueError:
					pass
				except TypeError:
					pass
				if prompt3=="n":
					prompt=""
					again=False
		post_data(prompt)
	 
	elif prompt=="l":
		iterations=int(input("How many times?"))
		prompt=""
		for i in range(iterations):
			enumerated=generate_data()
			random_index=rand_gen(enumerated,percent_limit)
			random_word=tokenizer.decode(int(enumerated[random_index][1][1].item()))
			#print(random_word)
			print(enumerated[random_index][0], round(enumerated[random_index][1][2].item(),4), round(enumerated[random_index][1][0].item(),4), random_word)
			post_data(random_word)
	
	else:
		post_data(prompt)