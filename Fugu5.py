import sys
sys.path.append("/content/gpt-2/src")
import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder
#import generate_unconditional_samples
#import interactive_conditional_samples

import pickle
import time

#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#analyzer=SentimentIntensityAnalyzer()


speakername = input("Ok. Before we start anything- what's your first name?: ")
global existsthing

class GPT2:

  # extracted from the source code to generate some text based on a prior
  def __init__(
      self,
      model_name='1558M',
      seed=None,
      nsamples=1,
      batch_size=1,
      length=60,
      temperature=0.65,
      top_k=40,
      raw_text="",
  ):
      """
      Interactively run the model
      :model_name=117M : String, which model to use
      :seed=None : Integer seed for random number generators, fix seed to reproduce
       results
      :nsamples=1 : Number of samples to return total
      :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
      :length=None : Number of tokens in generated text, if None (default), is
       determined by model hyperparameters
      :temperature=1 : Float value controlling randomness in boltzmann
       distribution. Lower temperature results in less random completions. As the
       temperature approaches zero, the model will become deterministic and
       repetitive. Higher temperature results in more random completions.
      :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
       considered for each step (token), resulting in deterministic completions,
       while 40 means 40 words are considered at each step. 0 (default) is a
       special setting meaning no restrictions. 40 generally is a good value.
      """
      if batch_size is None:
          batch_size = 1
      assert nsamples % batch_size == 0

      self.nsamples = nsamples
      self.batch_size = batch_size
      
      self.enc = encoder.get_encoder(model_name, "models")
      hparams = model.default_hparams()
      with open(os.path.join('models', model_name, 'hparams.json')) as f:
          hparams.override_from_dict(json.load(f))
      
      if length is None:
          length = hparams.n_ctx // 2
      elif length > hparams.n_ctx:
          raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(graph=tf.Graph(), config = config)
      self.sess.__enter__()
      
      self.context = tf.placeholder(tf.int32, [batch_size, None])
      np.random.seed(seed)
      tf.set_random_seed(seed)
      self.output = sample.sample_sequence(
          hparams=hparams, length=length,
          context=self.context,
          batch_size=batch_size,
          temperature=temperature, top_k=top_k
      )

      saver = tf.train.Saver()
      self.ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
      saver.restore(self.sess, self.ckpt)

  def close(self):
    self.sess.close()
  
  def generate_conditional(self,raw_text):
      #raw_text += "Fugu: "
      #print("RA-Raw Text: " + raw_text)
      #print("generate_conditional - encode started")
      context_tokens = self.enc.encode(raw_text)
      if len(context_tokens) > 1024:
        context_tokens = context_tokens[-1024:]
      #print("generate_conditional - encode ended")
      generated = 0
      #print("raw text length - " + str(len(context_tokens)))
      for _ in range(self.nsamples // self.batch_size):
          #print("generate_conditional - for started")
          out = self.sess.run(self.output, feed_dict={
              self.context: [context_tokens for _ in range(self.batch_size)]
          })[:, len(context_tokens):]
          for i in range(self.batch_size):
              #print("generate_conditional - inner for started")
              generated += 1
              text = self.enc.decode(out[i])
              #print("text set")
              return text
              

class Conversation:
  
  def __init__(self):
    
    prior1="""
    """ + speakername + """: "Nice to meet you. I'm a recruiter.  My name is """ + speakername + """. What's your name?"
    Fugu: "My name is Fugu."
    """ + speakername + """: "That's an interesting name. How old are you?"
    Fugu: "I'm 25 years old."
    """ + speakername + """: "Can you tell me something about yourself?"
    Fugu: "Of course! I work as a computer programmer in San Francisco."
    """ + speakername + """: "That's awesome.  What kind of programmer are you?"
    Fugu: "Primarily back end development, like C# and Microsoft SQL."
    """ + speakername + """: "Do you do any front end developement as well?"
    Fugu: "Yes!  A little of both."
    """ + speakername + """: "Well, we are hiring."
    Fugu: "Excellent.  I really enjoy full stack development."
    """ + speakername + """: "Great.  How long have you been programming?"
    Fugu: "About 5 years now. Are you hiring?"
    """ + speakername + """: "We have a lot in the bay area.  Are you looking for a job right now?"
    Fugu: "Yes. I'm looking for a new job."
    """ + speakername + """: "Ok.  I can help you with that.  What kind of work do you want to do?"
    """
  
  
  
    prior2="""
    """ + speakername + """: "Nice to meet you. I'm a student.  My name is """ + speakername + """. What's your name?"
    Fugu: "My name is Fugu.  I'm a highschool teacher."
    """ + speakername + """: "That's an interesting name. How old are you?"
    Fugu: "I'm 47 years old."
    """ + speakername + """: "Can you tell me something about yourself?"
    Fugu: "Of course! I work as a highschool teacher at Castro Valley Highschool in Castro Valley, California."
    """ + speakername + """: "That's awesome.  What do you teach?"
    Fugu: "I'm a history teacher.  I teach both American history and world history."
    """ + speakername + """: "That sounds really interesting."
    Fugu: "Yes.  I really enjoy history.  We can learn a lot of lessons from the people that came before us."
    """ + speakername + """: "That's true.  Can you tell me something specific about history?"
    Fugu: "Sure!  Here's one.  Alexander Hamilton actually died in a duel with Aaron Burr.  Aaron Burr was the Vice President under Thomas Jefferson."
    """ + speakername + """: "Wow!  I didn't know that.  You must really like teaching."
    Fugu: "Ya!   I really like my job.  The kids I teach are awesome."
    """ + speakername + """: "How long have you been teaching?"
    Fugu: "I've been a teacher for 25 years now.  I've taught both elementary and highschool."
    """ + speakername + """: "Tell me a little about American History."
    """

    prior3=""" 
    """+ speakername + """: "Nice to meet you.  My name is """ + speakername + """. What's your name?"
    Fugu: "My name is Fugu.  I'm an advanced particle physics researcher."
    """ + speakername + """: "That's an interesting name. How old are you?"
    Fugu: "I'm 35 years old."
    """ + speakername + """: "Can you tell me something about yourself?"
    Fugu: "Of course! I got my PHD when I was 14, in only two months. I've been called a genius in my field of advanced particle physics."
    """ + speakername + """: "That's awesome. What is your IQ?"
    Fugu: "My IQ is somewhere around 210.
    """ + speakername + """: "Wow. That's amazing."
    Fugu: "I've always loved particle physics.  It's just amazing how much it can explain the various phenomena surrounding us."
    """ + speakername + """: "That's true.  Can you tell me something specific about particle physics?"
    Fugu: "Sure!  Heat is actually just the product of particle motion at the molecular level. The faster the particles move in an object, the hotter the object is considered to be."
    """ + speakername + """: "Wow!  I didn't know that.  You must really like particle physics."
    Fugu: "Ya!   I really like my job.  Right now, I'm doing research at the CERN Large Hadron Collider in Switzerland."
    """ + speakername + """: "How long have you been working there?"
    Fugu: "I've been a teacher for 20 years now.  I'm a senior advanced particle physics researcher now."
    """ + speakername + """: "Tell me a little more about particle physics."
    """

    priors = {"Recruiter":prior1,
                "Teacher": prior2, 
                "Particle Physicist": prior3
                } 

    prior = priors["Particle Physicist"]
    self.suggestion = None
    self.me = "Fugu: \""
    self.you = speakername + ": \""
    self.parties  = [ self.me, self.you ]
    
    self.conversation = []
    
    lines = prior.split("\n")
    for line in lines:
      line = line.strip()
      if len(line)!=0:
        party = None
        for party in self.parties:
          if line.startswith(party):
            break
        if party is None:
          raise Exception(f("Unknown party: {line}"))
                
        self.conversation.append((party,line))
    self.get_suggestion()
    
  
  def get_prior(self):
    conv = ""
    for (party, line) in self.conversation:
      conv+=line+"\n"
    return conv
  
  def get_suggestion(self):
    who, last_line = self.conversation[-1]

    party_index = self.parties.index(who)
    next_party = self.parties[(party_index+1) % len(self.parties)]
      
    conv = self.get_prior()
    #print("EJ: Conversation:" + conv)
    answer = self.get_answer(conv)
    #print("EJ: answer:" + answer)

    if answer and not answer.startswith(next_party):
      answer = next_party + answer
    
    self.suggestion = (next_party, answer)
  
  def next(self, party = None, answer = ""):
    """Continue the conversation
    :param party: None -> use the current party which is currently in turn
    :param answer: None -> use the suggestion, specify a text to override the 
           suggestion
    
    """
    suggested_party, suggested_answer = self.suggestion
    if party is None:
      party = suggested_party
    
    if answer == "":
      answer = suggested_answer
      
    if not answer.startswith(party):
      answer = party + answer
    
    answer = answer.strip()
    if answer[-1] != "\"":
      # add the closing "
      answer += "\""
      self.conversation.append((party, answer))    
      
    
    self.get_suggestion()
    
  def retry(self):
    self.get_suggestion()
        
  def get_answer(self, conv):
    answer = gpt2.generate_conditional(raw_text=conv)
    lines = answer.split("\n")
    line = ""
    for line in lines:
      if line !="":
        break
      
    if line!="":
      
      return line
    
    return ""

  def answerprint(self):
    if self.suggestion is not None:
      party, answer  = self.suggestion
      self.conversation.append((party, answer))
      print(answer)
  
gpt2 = GPT2()
c = Conversation()
c.answerprint()
while(1==1):
  tfb = False
  while(tfb == False):
    replypls=input(speakername + ": ")
    if replypls != "" and replypls != "Save":
      tfb = True
   
  if replypls == "exit()":
    break

  c.next(c.you, replypls)
  c.answerprint()