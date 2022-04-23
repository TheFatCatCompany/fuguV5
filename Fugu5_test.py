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
      length=100,
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
      
      self.enc = encoder.get_encoder(model_name)
      self.hparams = model.default_hparams()
      with open(os.path.join('models', model_name, 'hparams.json')) as f:
          self.hparams.override_from_dict(json.load(f))

      if length is None:
          length = self.hparams.n_ctx // 2
      elif length > self.hparams.n_ctx:
          raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

      self.sess = tf.Session(graph=tf.Graph())
      self.sess.__enter__()
      
      self.context = tf.placeholder(tf.int32, [batch_size, None])
      np.random.seed(seed)
      tf.set_random_seed(seed)
      self.output = sample.sample_sequence(
          hparams=self.hparams, length=length,
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
      #print("generate_conditional - encode started")
      context_tokens = self.enc.encode(raw_text)
      #print("Last token is " + self.enc.decode(context_tokens[-2:]))
      if len(context_tokens) > 1024:
        context_tokens = context_tokens[-1024:]
      #print("generate_conditional - encode ended")
      generated = 0

      # self.output = sample.sample_sequence(
      #     hparams=self.hparams, length=5,
      #     context=self.context,
      #     batch_size=1,
      #     temperature=.2, top_k=40
      # )


      print("raw text length - " + str(len(context_tokens)))
      for _ in range(self.nsamples // self.batch_size):
          #print("generate_conditional - for started")
          out = self.sess.run(self.output, feed_dict={
              self.context: [context_tokens for _ in range(self.batch_size)]
          })[:, len(context_tokens):]
          encodedNewLine = self.enc.encode("\n")
          print("new line is " + str(encodedNewLine))
          for i in range(self.batch_size):
              #print("generate_conditional - inner for started")
              generated += 1
              text = self.enc.decode(out[i])
              #print("text set")
              return text
              

class Conversation:
  
  def __init__(self, prior = None):
    if prior is None:
      prior="""
      """ + speakername + """: "Nice to meet you. What's your name?"
      Fugu: "My name is Fugu."
      """ + speakername + """: "That's an interesting name. How old are you?"
      Fugu: "I'm 25 years old."
      """ + speakername + """: "Can you tell me something about yourself?"
      Fugu: "Of course! I work as a computer programmer in San Francisco."
      """ + speakername + """: "That's awesome.  What kind of programmer are you?"
      Fugu: "Primarily back end development, like C# and Microsoft SQL."
      """ + speakername + """: "Do you do any front end developement as well?"
      Fugu: "A little.  I've messed around with Angular and javascript, and a little css here and there."
      """ + speakername + """: "Sweet.  Are you looking for a job right now?"
      """
    
    
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