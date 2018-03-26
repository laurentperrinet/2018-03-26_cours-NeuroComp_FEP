#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
__author__ = "Laurent Perrinet INT - CNRS"
__licence__ = 'BSD licence'
DEBUG = True
DEBUG = False
"""
Course on Computational Neuroscience : Bayes

"""
figpath_bcp = 'figures' # os.path.join(home, 'pool/ANR-REM/ASPEM_REWARD/AnticipatorySPEM/2017-11-24_Poster_GDR_Robotique/figures')


import sys
#print(sys.argv)
tag = sys.argv[1].split('.')[0]
slides_filename = sys.argv[1]
print('😎 Welcome to the script generating the slides for ', tag)
YYYY = int(tag[:4])
MM = int(tag[5:7])
DD = int(tag[8:10])

import os
home = os.environ['HOME']

# see https://github.com/laurentperrinet/slides.py
from slides import Slides
height_px = 80
height_ratio = .9

meta = dict(
 embed = True,
 draft = DEBUG, # show notes etc
 width= int(1600*height_ratio),
 height= int(1000*height_ratio),
 #margin= 0.1618,#
 margin= 0.1,#
 #reveal_path = 'https://s3.amazonaws.com/hakim-static/reveal-js/',
 reveal_path = 'http://cdn.jsdelivr.net/reveal.js/3.0.0/',
 #reveal_path = 'https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.4.1/',
 #theme='night',
 #theme='sky',
 #theme='black',
 #theme='White',
 theme='simple',
 author='',
 author_link='<a href=mailto:laurent.perrinet@univ-amu.fr>Laurent Udo Perrinet, INT</a>',
 title="""Probabilities, Bayes and the Free-energy principle""",
 short_title='',
 location='R+1, INT',
 conference='PhD program in  Neuroscience, Marseille <BR> March 27th, 2018 <BR> ⓦ <a href="https://invibe.net/LaurentPerrinet/Presentations/2018-03-26_cours-NeuroComp">https://invibe.net/LaurentPerrinet/Presentations/2018-03-26_cours-NeuroComp</a><BR>Presentation made with <a href="http://laurentperrinet.github.io/slides.py/index.html">slides.py</a>',
 YYYY = YYYY,
 MM = MM,
 DD = DD,
 tag = tag,
 url = 'http://invibe.net/LaurentPerrinet/Presentations/' + tag,
 abstract="""
""",
 sections= ['Problem statement',
    'Probabilities and Bayesian inference',
    'Practical example: How to decode neural activity?',
    'Variational Inference and the Free-energy principle',
    #'Active inference, EMs & oculomotor delays',
    'Take-home message']
 )

print("""
#acl All:read

= {title}  =

 Quoi:: {conference}
 Qui::
 Quand:: {DD}/{MM}/{YYYY}
 Où:: {location}
 Support visuel:: http://blog.invibe.net/files/{tag}.html

== reference ==
{{{{{{
#!bibtex
@inproceedings{{{tag},
	Author = "{author}",
    Booktitle = "{conference}",
    Title = "{title}",
	Url = "{url}",
	Year = "{YYYY}",
}}
}}}}}}
## add an horizontal rule to end the include

----
<<Include(BibtexNote)>>
----
TagTalks TagYear18 TagPublic

""".format(**meta) )

do_section = [True] * (len(meta['sections']) + 2)
i_section = 0
s = Slides(meta)
s.meta['Acknowledgements'] ="""<h3>Acknowledgements:</h3>
   <ul>
    <li>Laurent Pezard & Demian Battaglia, INS</li>
    <li>Anna Montagnini, Nicole Malfait & Frédéric Chavane, INT</li>
    <li>Stéphanie Ouine, PhD program</li>
    </ul>
    <BR>

"""
# <li>Mina Aliakbari Khoei, Guillaume Masson and Anna Montagnini - FACETS-ITN Marie Curie Training</li>

bgcolor = "white"
height_ratio = 1.2

if do_section[i_section]:
    s.open_section()
    #################################################################################
    ## Intro - 5''
    #################################################################################
    #################################################################################
    figpath = os.path.join(home, 'nextcloud/libs/slides.py/figures/')
    s.add_slide(hide=True, content=s.content_figures(
        [os.path.join(figpath, 'mire.png')], cell_bgcolor=bgcolor,
        height=s.meta['height']*height_ratio),
        #image_fname=os.path.join(figpath, 'mire.png'),
        notes="""
Check-list:
-----------

* (before) bring miniDVI adaptors, AC plug, remote, pointer
* (avoid distractions) turn off airport, screen-saver, mobile, sound, ... other running applications...
* (VP) open monitor preferences / calibrate / title page
* (timer) start up timer
* (look) @ audience
 """)
    intro = """
    <h2 class="title">{title}</h2>
    <h3>{author_link}</h3>
    """.format(**meta)
    intro += s.content_figures(
    [os.path.join(figpath, "troislogos.png")], cell_bgcolor=bgcolor,
    height=s.meta['height']*.2, width=s.meta['height']*.8)
    intro += """
    {Acknowledgements}
    <h4>{conference}</h4>
    """.format(**meta)
    s.add_slide(content=intro,
        notes="""
* (CONF) Hello, I am Laurent Perrinet, I will guide you through this introduction to probabilities, bayesian inference and the free-energy principle

* (SHOW AUTHOR) I am interested in the link between the neural code and the structure of the world. in particular, for vision, I am researching the relation between the functional organization (anatomy and activity) of low-level visual areas and the structures that appear to be in natural scenes, that is of the images that hit the retina and which are relevant to us.

* (ACKNO) From the head on, I wish to thanks people who collaborated on this endeavour, ....

* (SHOW TITLE) of interest for biologists to understand what the neural activity (or behavior) they record relates to something relevant (a function, a particular object)

 """)

    figpath = 'figures'

    figname = os.path.join(figpath_bcp, 'qr.png')
    if not os.path.isfile(figname):
        # https://pythonhosted.org/PyQRCode/rendering.html
        # pip3 install pyqrcode
        # pip3 install pypng
        import pyqrcode as pq

        code = pq.create(meta['url'])
        code.png(figname, scale=5)

    s.add_slide(content=s.content_figures([figname], cell_bgcolor=bgcolor, height=s.meta['height']*height_ratio),
    notes=""" All the material is available online - please flash this QRcode """)

    # open questions:
    #figpath = '../2017-03-06_cours-NeuroComp_intro/figures'
    bib =  '(see this <a href=" http://viperlib.york.ac.uk/areas/15-anatomy-physiology/contributions/2032-hubel-and-wiesel">viperlib</a> page)'

    for fname in ['scientists.jpg']:
        s.add_slide(content=s.content_figures(
           [os.path.join(figpath, fname)], cell_bgcolor=bgcolor,
           title=meta['sections'][i_section], height=s.meta['height']*height_ratio) + bib,
            notes="""
* (INTRO) Indeed, a conundrum in visual neuroscience, and neuroscience in general, is to infer the causes that underly the firing of any given neural cell. in visual neuroscience, this is conceptualized by the term of receptive field [[---a concept which dates back to Sherrington and to the scratch reflex of the dog---]] and which corresponds to the set of visual stimuli that causes the firing of any given neuron, this set being guessed by changing the parameters of the visual stimulation,
 * to illustrate this methodology, I love to see this picture of David Hubel and Torsten Wiesel performing their Nobel Prize-winning experiments on area V1.  ( source: http://viperlib.york.ac.uk/areas/15-anatomy-physiology/contributions/2032-hubel-and-wiesel ) -
 """)

    s.add_slide(content="""
        <video controls width=99%/>
          <source type="video/mp4" src="{}">
        </video>
        <BR>
        """.format(s.embed_video(os.path.join(figpath, 'ComplexDirSelCortCell250_title.mp4'))) + bib,
    notes="""
on the video, they characterize a complex cell from area V1 by manipulating the visual stimulation's parameters: central position, orientation of a bar, direction, ...
* -> As a consequence,  Neurons are often characterized using simple stimuli like bars or grating

    """)

    for fname in ['en', 'de']:
        s.add_slide(content=s.content_figures(
        [os.path.join(figpath, fname + 'coding_problem.png')], cell_bgcolor=bgcolor,
        title='Summary: the encoding / decoding problem', height=s.meta['height']*height_ratio),
         notes="""

ultimate goal = guess the stimulus (the hidden variables $\theta$) from a particular neuronal response (an observed spike raster Y) given a belief on response model(using estimated statistics such as the mean (corresponding to the tuning function) parametrized by $\theta$ : $f_{i}(\theta)$).

but in order to perform this inverse problem it is necessarary to put all information in the system, the "prior"

""")
    bib = s.content_bib("LT Maloney", "2002", "Journal of Vision", url="http://jov.arvojournals.org/article.aspx?articleid=2121565")

    bib2 = s.content_bib("Ernst & Bülthoff", "2004", "Trends in Cognitive Sciences", url="http://dx.doi.org/10.1016/j.tics.2004.02.002")

    for fname, bib in [('dotsconvex.jpeg', ''), ('dotsconcave.jpeg', ''), ('hollow_mask.jpg', ''), ('jov-2-6-6-fig002.jpeg', bib), ('7b90ecdea9.jpg', bib2) ]:
        s.add_slide(content=s.content_figures(
           [os.path.join(figpath, fname)], cell_bgcolor=bgcolor,
           title='Examples of Bayesian mechanisms in perception', height=s.meta['height']*height_ratio) + bib,
            notes="""
Let's see some effects of priors in perception

    * convex vs concave
    * hollow mask
    * cardinals
    * cross-modal integration

    """)

    s.close_section()

i_section += 1
if do_section[i_section]:
    s.open_section()
    title = meta['sections'][i_section]
    s.add_slide_outline(i_section)

    bib =  'Let''s <a href="https://try.jupyter.org/">try</a> this using this <a href="http://blog.invibe.net/posts/2017-03-09_probabilities.ipynb">notebook</a> (<a href="http://blog.invibe.net/posts/2017-03-09_probabilities.html">solution</a>)'

    s.add_slide(content=s.content_figures(
           [os.path.join(figpath, 'prob-dice.png')], cell_bgcolor=bgcolor,
           title=meta['sections'][i_section], height=s.meta['height']*height_ratio) + bib,
            notes="""# Bayes : travelling back to the feature space

probability:  Bayesians interpret a probability as a measure of belief, or confidence, of an event occurring.

* p(s)
* p(r|s)
* bayes p(s|r) . p(s) =  p(r|s) . p(s)
* cost function

            """)

    for fname, note in [('1D-rain', """


 Visualizing Probability Distributions

Before we dive into bayesian inference, let’s think about how we can visualize simple probability distributions. We’ll need this later on, and it’s convenient to address now. As a bonus, these tricks for visualizing probability are pretty useful in and of themselves!

I’m in Marseille. Sometimes it rains, but mostly there’s sun! Let’s say it’s sunny 75% of the time. It’s easy to make a picture of that:"""),

                        ('1D-coat', """

Most days, I wear a t-shirt, but some days I wear a coat. Let’s say I wear a coat 38% of the time. It’s also easy to make a picture for that!

What if I want to visualize both at the same time? We’ll, it’s easy if they don’t interact – if they’re what we call independent. For example, whether I wear a t-shirt or a raincoat today doesn’t really interact with what the weather is next week. We can draw this by using one axis for one variable and one for the other:

                        """), (

                        '2D-independent-rain', """

Notice the straight vertical and horizontal lines going all the way through. That’s what independence looks like! 1 The probability I’m wearing a coat doesn’t change in response to the fact that it will be raining in a week. In other words, the probability that I’m wearing a coat and that it will rain next week is just the probability that I’m wearing a coat, times the probability that it will rain. They don’t interact.

                        """), (

                        '2D-dependant-rain-squish', """

When variables interact, there’s extra probability for particular pairs of variables and missing probability for others. There’s extra probability that I’m wearing a coat and it’s raining because the variables are correlated, they make each other more likely. It’s more likely that I’m wearing a coat on a day that it rains than the probability I wear a coat on one day and it rains on some other random day.

Visually, this looks like some of the squares swelling with extra probability, and other squares shrinking because the pair of events is unlikely together:

But while that might look kind of cool, it’s isn’t very useful for understanding what’s going on.


                        """), ('2D-factored-rain-arrow', """

Instead, let’s focus on one variable like the weather. We know how probable it is that it’s sunny or raining. For both cases, we can look at the conditional probabilities. How likely am I to wear a t-shirt if it’s sunny? How likely am I to wear a coat if it’s raining?

There’s a 25% chance that it’s raining. If it is raining, there’s a 75% chance that I’d wear a coat. So, the probability that it is raining and I’m wearing a coat is 25% times 75% which is approximately 19%. The probability that it’s raining and I’m wearing a coat is the probability that it is raining, times the probability that I’d wear a coat if it is raining. We write this:

p(rain,coat)=p(rain)⋅p(coat | rain)

This is a single case of one of the most fundamental identities of probability theory:

p(x,y)=p(x)⋅p(y|x)

We’re factoring the distribution, breaking it down into the product of two pieces. First we look at the probability that one variable, like the weather, will take on a certain value. Then we look at the probability that another variable, like my clothing, will take on a certain value conditioned on the first variable.


                        """), ('2D-factored1-clothing-B', """

The choice of which variable to start with is arbitrary. We could just as easily start by focusing on my clothing and then look at the weather conditioned on it. This might feel a bit less intuitive, because we understand that there’s a causal relationship of the weather influencing what I wear and not the other way around… but it still works!

Let’s go through an example. If we pick a random day, there’s a 38% chance that I’d be wearing a coat. If we know that I’m wearing a coat, how likely is it that it’s raining? Well, I’m more likely to wear a coat in the rain than in the sun, but rain is kind of rare in California, and so it works out that there’s a 50% chance that it’s raining. And so, the probability that it’s raining and I’m wearing a coat is the probability that I’m wearing a coat (38%), times the probability that it would be raining if I was wearing a coat (50%) which is approximately 19%.

p(rain,coat)=p(coat)⋅p(rain | coat)

This gives us a second way to visualize the exact same probability distribution.


Note that the labels have slightly different meanings than in the previous diagram: t-shirt and coat are now marginal probabilities, the probability of me wearing that clothing without consideration of the weather. On the other hand, there are now two rain and sunny labels, for the probabilities of them conditional on me wearing a t-shirt and me wearing a coat respectively.

(You may have heard of Bayes’ Theorem. If you want, you can think of it as the way to translate between these two different ways of displaying the probability distribution!)

                        """)]:
        s.add_slide(content=s.content_figures(
           [os.path.join(figpath, 'prob-' + fname + '.png')], cell_bgcolor=bgcolor,
           title=meta['sections'][i_section], height=s.meta['height']*height_ratio)+ '(see <a href="http://colah.github.io/posts/2015-09-Visual-Information/">http://colah.github.io/posts/2015-09-Visual-Information/</a>)',
            notes=note)



    s.close_section()

i_section += 1
if do_section[i_section]:
    s.open_section()
    title = meta['sections'][i_section]
    s.add_slide_outline(i_section)


    for fname, note in [('1', """

## The definition of a model for the inter-trial variability of the response

Ex: Poisson model, only one parameter : the mean $\mu_0$= 10 spikes


"""),
                        ('2', """



By the definition of a Poisson distribution, the spike count follows a homogenous Poisson process with a rate parameter ${\lambda}$ (the expected number of spikes that occur per unit of time). If $k$ is the observed number of spikes fired by a neuron in response to a stimulus in a time window $\Delta t$,  the probability of such an observation is given by a Poisson distribution of parameter $\mu_0=\lambda \Delta t$, which gives the expected number of spikes per sample in the considered time window:
\begin{equation}
P(k) = \frac{\mu_0 ^{k}e^{-\mu_0}}{k!}
\end{equation}

 * $\mu_0$ corresponds to the average value of possible spike counts generated by a Poisson model of parameter $\mu_0$.


                    """),
                                            ('3', """

- "Describing neurons’ activity" +  "Poisson Distribution"

2. Describing ‘the noise’
 • Beyond describing only the mean spike count ...
•To model the statistics of the response (one trial), we can use tools from probability theory: stochastic (random) processes.
• The spike count r on one trial is considered as a random variable.
• The probability of getting each outcome (n=1,2 .., 10, 50 spikes) is given by a probability distribution for which we want to find a suitable model.
• To do that, we use known statistics of n: the mean <n>=f(s) and 2d order statistics (variance, correlations).



                                        """),
                                                                ('4', """

Poisson Distribution
• probability of a spike count (positive integer -- discrete probability distribution) occurring in a fixed period of time, knowing average spike count f(s)
• The assumption is that the generation of each spike (and its stochasticity) is independent of all other spikes



                                                            """),
                                                                                    ('5', """


*  "1. Modeling the average firing rate <r(s)>"  + gaussian + single cell tuning curves vs population response
*  "“Tuning Curve + Noise” Population Model" + Jazayeri



                                                                                """)]:
        s.add_slide(content=s.content_figures(
           [os.path.join(figpath, 'neural_activity_' + fname + '.png')], cell_bgcolor=bgcolor,
           title=meta['sections'][i_section], height=s.meta['height']*height_ratio) ,
            notes=note)

    Jazayeri_bib = s.content_bib("Mehrdad Jazayeri & J Anthony Movshon ", "2007", "Nature Neuroscience", url="http://www.nature.com/neuro/journal/v9/n5/abs/nn1691.html")

    for fname, note in [('1', """


Figure 1 Computing the log likelihood function in a feedforward network. At its input (bottom), a stimulus, elicits n1, n2, y , nN spikes in the sensory representation. The response of each neuron multiplied by the logarithm of its own tuning curve, log[fi], gives the contribution of that neuron to the log likelihood function. Adding the contribution of individual neurons (shown for two example stimulus values in orange and green) gives the overall log likelihood function, log L(y) for all values of y that could have elicited this pattern of responses. Here, the orange point at the peak of the log likelihood function indicates the most likely stimulus.

"""),
                        ('2', """


Figure 2 Computing likelihood for the direction of motion. (a) A random-dot stimulus (bottom) activates a set of directionally tuned neurons in area MT. The smooth curves represent neuronal tuning curves, and small circles show the noise-perturbed population response on a particular trial. To represent likelihood, we recoded the sensory signals by weighting the inputs from the population of tuned ‘encoding’ neurons. For the example shown, the correct weighting function has a cosinusoidal form, and the weighted signals converge to an output neuron representing log likelihood for a leftward direction. (b) Same as a, except here the output layer consists of an ensemble of neurons. The weighted signals converge to this output layer where the neurons represent the log likelihood for all possible directions, the likelihood function. Here, at the output, the average likelihood profile is shown; the colored points represent the average likelihoods of four example directions. The peak of the average likelihood function—the expected maximum-likelihood estimate of the stimulus direction—is shown as orange.


##  Maximum likelihood decoding.

The decoding algorithm consists of maximizing the posterior probability $P({\theta}|Y)$ as a function of the estimated direction ${\theta}$, given a distribution hypothesis:
* The evidence term $P(Y)$ is a normalization term independent of ${\theta}$ $\to P(Y)$=cst
* There is no prior knowledge on ${\theta}$ (such that $ \forall \theta_1, \theta_2$,  $P(\theta_1)$ =  $P(\theta_2)$)

Thus, maximizing the posterior $P(\theta|Y)$ under the Poisson hypothesis is equivalent to maximizing the following likelihood function:
\begin{equation}
    L(\theta) = P(Y|{\theta}) = \Pi ^N  _{i=1}  \frac{f_i({\theta}) ^{k_i}e^{-f_i({\theta})}}{k_i!}
\end{equation}

In practice, It is often the log-likelihood function that is considered:

\begin{equation}
    LL(\theta) = log(P(Y|{\theta})) = \sum_{i=1}^N{k_i\log[f_{i}(\theta)]}-\sum_{i=1}^N{f_{i}(\theta)}- \sum_{i=1}^N \log[{k_i!}]
\end{equation}

In the end:

\begin{equation}
    LL(\theta) = \sum_{i=1}^N{k_i\log[f_{i}(\theta)]} - \log(Z)
\end{equation}


                    """)]:
        s.add_slide(content=s.content_figures(
           [os.path.join(figpath, 'Jazayeri07optimal_figure' + fname + '.png')], cell_bgcolor=bgcolor,
           title='Optimal representation of sensory information', height=s.meta['height']*height_ratio) + Jazayeri_bib,
            notes=note)

    s.close_section()



i_section += 1
if do_section[i_section]:
    s.open_section()
    title = meta['sections'][i_section]
    s.add_slide_outline(i_section, notes="""

Great; but now, what if I want to learn the right parameters instead of scanning all of them?
A solution lies in the Free-energy princple from Karl Friston, and I will give here an introduction to this technique
""")

    karl_bib = s.content_bib("Friston", "2010", "Nat Neuro Reviews", url="http://www.nature.com/nrn/journal/v11/n2/abs/nrn2787.html")

    s.add_slide(content=s.content_figures(
       [os.path.join(figpath, 'friston10c_fig4.png')], cell_bgcolor=bgcolor,
       title=title, height=s.meta['height']*height_ratio) + karl_bib,
          notes="""

""")


    bib =  'Let''s <a href="https://try.jupyter.org/">try</a> this using this <a href="http://blog.invibe.net/posts/2017-01-15-bogacz-2017-a-tutorial-on-free-energy.ipynb">notebook</a> (<a href="http://blog.invibe.net/posts/2017-01-15-bogacz-2017-a-tutorial-on-free-energy.html">solution</a>)'

    bogacz_bib = s.content_bib("Bogacz", "2017", "Journal of Mathematical Psychology", url="http://dx.doi.org/10.1016/j.jmp.2015.11.003")

    s.add_slide(content=s.content_figures(
       [os.path.join(figpath, 'bogacz-fig3.jpg')], cell_bgcolor=bgcolor,
       title=title, height=s.meta['height']*height_ratio) + bogacz_bib,
          notes="""




I enjoyed reading "A tutorial on the free-energy framework for modelling perception and learning" by *Rafal Bogacz*, which is freely available [here](http://www.sciencedirect.com/science/article/pii/S0022249615000759). In particular, the author encourages to replicate the results in the paper. He is himself giving solutions in matlab, so I had to do the same in python all within a notebook...

          A tutorial on the free-energy framework for modelling perception and learning
        http://dx.doi.org/10.1016/j.jmp.2015.11.003

 * online
 * complex priors

The architecture of the model performing simple perceptual inference. Circles denote neural “nodes”, arrows denote excitatory connections, while lines ended with circles denote inhibitory connections. Labels above the connections encode their strength, and lack of label indicates the strength of 1. Rectangles indicate the values that need to be transmitted via the connections they label.

""")



    for fname, note in [('1', """

First, let's see the application of Bayes theorem
We start by considering in this section a simple perceptual problem in which a value of a single variable has to be inferred from a single observation. To make it more concrete, consider a simple organism that tries to infer the size or diameter of a food item, which we denote by $v$, on the basis of light intensity it observes. Let us assume that our simple animal has only one light sensitive receptor which provides it with a noisy estimate of light intensity, which we denote by $u$. Let g
 denote a non-linear function relating the average light intensity with the size. Since the amount of light reflected is related to the area of an object, in this example we will consider a simple function of $g(v)=v^2$. Let us further assume that the sensory input is noisy—in particular, when the size of food item is v, the perceived light intensity is normally distributed with mean g(v)
, and variance $Σ_u$ (although a normal distribution is not the best choice for a distribution of light intensity, as it includes negative numbers, we will still use it for a simplicity):
$$
p(u|v)=f(u;g(v),Σu).
$$
In $f(x;μ,Σ) $ denotes the density of a normal distribution with mean μ  and variance Σ

Due to the noise present in the observed light intensity, the animal can refine its guess for the size v
 by combining the sensory stimulus with the prior knowledge on how large the food items usually are, that it had learnt from experience. For simplicity, let us assume that our animal expects this size to be normally distributed with mean $v_p$  and variance $Σ_p$  (subscript p stands for “prior”), which we can write as:
$$
p(v)=f(v;vp,Σp).
$$

Let us now assume that our animal observed a particular value of light intensity, and attempts to estimate the size of the food item on the basis of this observation. We will first consider an exact solution to this problem, and illustrate why it would be difficult to compute it in a simple neural circuit. Then we will present an approximate solution that can be easily implemented in a simple network of neurons.

## Exact solution

To compute how likely different sizes $v$ are given the observed sensory input $u$, we could use Bayes’ theorem:
$$
p(v|u)=p(v)p(u|v)p(u).
$$

Term $p(u)$  in the denominator of equation is a normalization term, which ensures that the posterior probabilities of all sizes $p(v|u)$  integrate to 1:

$$
p(u)=∫p(v)p(u|v)dv.
$$

The integral in the above equation sums over the whole range of possible values of $v$, so it is a definite integral, but for brevity of notation we do not state the limits of integration in this and all other integrals in the paper.

Now combining Eqs. we can compute numerically how likely different sizes are given the sensory observation. For readers who are not familiar with such Bayesian inference we recommend doing the following exercise now.

Exercise 1.

Assume that our animal observed the light intensity  $u=2$, the level of noise in its receptor is  $Σ_u=1$, and the mean and variance of its prior expectation of size are  $v_p=3$ and  $Σ_p=1$. Write a computer program that computes the posterior probabilities of sizes from  0.01  to  5, and plots them.

"""),
                        ('2', """

By
## Finding the most likely feature value

Instead of finding the whole posterior distribution  p(v|u) , let us try to find the most likely size of the food item  v  which maximizes  p(v|u) . We will denote this most likely size by  ϕ , and its posterior probability density by  p(ϕ∣∣u) . It is reasonable to assume that in many cases the brain represents at a given moment of time only most likely values of features. For example in case of binocular rivalry, only one of the two possible interpretations of sensory inputs is represented.

We will look for the value  ϕ  which maximizes  p(ϕ∣∣u) . According to Eq. (4), the posterior probability  p(ϕ∣∣u)  depends on a ratio of two quantities, but the denominator  p(u)  does not depend on  ϕ . Thus the value of  ϕ  which maximizes  p(ϕ∣∣u)  is the same one which maximizes the numerator of Eq. (4). We will denote the logarithm of the numerator by  F , as it is related to the negative of free energy (as we will describe in Section  3):

$$
F=lnp(ϕ)+lnp(u∣∣ϕ).
$$
 the definition of this model, one may derive a learning rule to estimate the best $u$

Let's define $F = \log( p(v |u) )$ after some derivation, we get

$$
\frac{\partial F}{\partial v} = \frac{v - v_p}{\Sigma_p} + \dot{g}(v) \cdot \frac{u - g(v)}{\Sigma_u}
$$


In the above equation we used the property of logarithm  $ln(ab)=lna+lnb$. We will maximize the logarithm of the numerator of Eq. (4), because it has the same maximum as the numerator itself as  ln  is a monotonic function, and is easier to compute as the expressions for  p(u∣∣ϕ)  and  p(ϕ)  involve exponentiation.

To find the parameter  ϕ  that describes the most likely size of the food item, we will use a simple gradient ascent: i.e. we will modify  ϕ  proportionally to the gradient of  F , which will turn out to be a very simple operation. It is relatively straightforward to compute  F  by substituting Eqs. (1), (2) ;  (3) into Eq. (6) and then to compute the derivative of  F  (TRY IT YOURSELF).

equation(7)
F=lnf(ϕ;vp,Σp)+lnf(u;g(ϕ),Σu)=ln[12πΣp√exp(−(ϕ−vp)22Σp)]+ln[12πΣu√exp(−(u−g(ϕ))22Σu)]=ln12π√−12lnΣp−(ϕ−vp)22Σp+ln12π√−12lnΣu−(u−g(ϕ))22Σu=12(−lnΣp−(ϕ−vp)2Σp−lnΣu−(u−g(ϕ))2Σu)+C.
Turn MathJax off

We incorporated the constant terms in the 2nd line above into a constant  C . Now we can compute the derivative of  F  over  ϕ :

equation(8)
∂F∂ϕ=vp−ϕΣp+u−g(ϕ)Σug′(ϕ).
Turn MathJax off

In the above equation we used the chain rule to compute the second term, and  g′(ϕ)  is a derivative of function  g  evaluated at  ϕ , so in our example  g′(ϕ)=2ϕ . We can find our best guess  ϕ  for  v  simply by changing  ϕ  in proportion to the gradient:

equation(9)
ϕ̇ =∂F∂ϕ.
Turn MathJax off

In the above equation  ϕ̇   is the rate of change of  ϕ  with time. Let us note that the update of  ϕ  is very intuitive. It is driven by two terms in Eq. (8): the first moves it towards the mean of the prior, the second moves it according to the sensory stimulus, and both terms are weighted by the reliabilities of prior and sensory input respectively.

Now please note that the above procedure for finding the approximate distribution of distance to food item is computationally much simpler than the exact method presented at the start of the paper. To gain more appreciation for the simplicity of this computation we recommend doing the following exercise.

Exercise 2.

Write a computer program finding the most likely size of the food item   ϕ for the situation described in   Exercise  1. Initialize   ϕ=vp , and then find its values in the next  5  time units (you can use Euler’s method, i.e. update   $ϕ(t+Δt)=ϕ(t)+Δt∂F/∂ϕ$ with   t=0.01$ ).

Fig. 2(a) shows a solution to Exercise 2. Please notice that it rapidly converges to the value of  $ϕ≈1.6 $, which is also the value that maximizes the exact posterior probability  $p(v|u)$  shown in Fig. 1.



                    """),
                                            ('3', """

one may also learn the variance

The model parameters can be hence optimized by modifying them proportionally to the gradient of  $F$. It is straightforward to find the derivatives of  F  over  $v_p$, $Σ_p$  and  $Σ_u$:

$$
\frac{∂F}{∂v_p}=\frac{ϕ−v_p}{Σ_p}
$$
$$
\frac{∂F}{∂Σp}=\frac 1 2 ( \frac {(ϕ−v_p)^2} {Σ^2_p}−\frac{1}{Σ_p} )
$$
$$
\frac{∂F}{∂Σu}=\frac 1 2 ( \frac {(u−g(ϕ))^2}{Σ^2_u}−\frac{1}{Σ_u}.
$$


> Simulate learning of variance  $Σ_i$ over trials. For simplicity, only simulate the network described by Eqs. (59)– (60), and assume that variables  ϕ are constant. On each trial generate input  $ϕ_i$ from a normal distribution with mean  5  and variance  2, while set  $g_i(ϕ_i+1)=5$ (so that the upper level correctly predicts the mean of  $ϕ_i$). Simulate the network for  20  time units, and then update weight  $Σ_i$ with learning rate  $α=0.01$. Simulate  1000  trials and plot how $Σ_i$ changes across trials.
​

                                        """)]:
        s.add_slide(content=s.content_figures(
           [os.path.join(figpath, 'bogacz_' + fname + '.png')], cell_bgcolor=bgcolor,
           title=meta['sections'][i_section], height=s.meta['height']*height_ratio) + bib,
            notes=note)

    s.close_section()


#
# i_section += 1
# if do_section[i_section]:
#     s.open_section()
#     ############################################################################
#     ## Free-energy / delays - 15''
#     ############################################################################
#     ############################################################################
#     title = meta['sections'][i_section]
#     s.add_slide_outline(i_section)
#
#     figpath = os.path.join(home, 'quantic/science/2017-01_LACONEU/figures/')
#     figpath = 'figures/'
#
#     s.add_slide(content=s.content_figures(
#        [os.path.join(figpath, 'tsonga.png')], bgcolor="white",
#        height=s.meta['height']*height_ratio),
#        notes="""
#     * ... As a consequence, for a tennis player ---here (highly trained) Jo-Wilfried Tsonga at Wimbledon--- trying to intercept a passing-shot ball at a (conservative) speed of $20~m.s^{-1}$, the position sensed on the retinal space corresponds to the instant when its image formed on the photoreceptors of the retina and reaches our hypothetical motion perception area behind:
#
#      """)
#     s.add_slide(content=s.content_figures(
#        [os.path.join(figpath, 'figure-tsonga.png')], bgcolor="white",
#        height=s.meta['height']*height_ratio),
#        #image_fname=os.path.join(figpath, 'figure-tsonga.png'), embed=False,
#            notes="""
#
# * and at this instant, the sensed physical position is lagging behind (as represented here by $\tau_s \cdot v 1~m$ ), that is, approximately at $45$ degrees of eccentricity (red dotted line),
#
# * while the  position at the moment of emitting the motor command will be $.8~m$ ahead of its present physical position ($\tau_m \cdot v$).
#
# * As a consequence, note that the player's gaze is directed to the ball at its **present** position (red  line), in anticipatory fashion. Optimal control directs action (future motion  of the eye) to the expected position (red dashed line) of the ball in the  future --- and the racket (black dashed line) to the expected position of the  ball when motor commands reach the periphery (muscles). This is obviously an interesting challenge for modelling an optimal control theory.
#
#  """)
# #
# #     s.add_slide(content=s.content_figures(
# #        [os.path.join(figpath, 'figure-tsonga-AB.png')], bgcolor="white",
# #        height=s.meta['height']*height_ratio),
# #        #image_fname=os.path.join(figpath, 'figure-tsonga-AB.png'),  embed=False,
# #            notes="""
# #     * As such, during this talk I will first outline one solution to explain how we may perceive a visual motion while compensating for sensory delays
# #
# #     * knowing this solution, we will then use it in AI to propose a complete solution for "eye movements & oculomotor delays"
# #
# #      """)
# #
# #     figpath = 'figures/'
# #     # karl_bib = s.content_bib("Friston", "2010", "Nat Neuro Reviews")
# #
# #     s.add_slide(content=s.content_figures(
# #       [os.path.join(figpath, 'figure-tsonga-B.png')], bgcolor="white",
# #       height=s.meta['height']*height_ratio),
# #       #image_fname=os.path.join(figpath, 'figure-tsonga-AB.png'),
# #        notes="""
# #
# # In that order, we will now show how to include oculomotor delays
# #
# # and include constraints from EMs as a model for a generic model of decision making
# #
# # thanks to a one year sabbatical visit at Karl Friston's lab in London at the WTCI UCL and in collaboration with Rick Adams, I have had the chance to collaborate in the ellaboration of a series of studies on Active inference and EMs:
# #
# # """)
#
#     # s.add_slide(content=s.content_figures(
#     # [os.path.join(figpath, 'friston10c_fig4.png')], bgcolor="white",
#     # title=title, height=s.meta['height']*.6) + karl_bib,
#     #    notes="""
#     #
#     # As we now all know, the ...
#     #
#     # Unification des theories computationnelles par la minimisation de l'energie libre (MEL).
#     # Cette figure extraite de {Friston10c} represente la place central du principe de MEL dans l'ensemble des theories computationnelles. En particulier, on peut noter que les principes que nous avons detailles plus haut dans les chapitres precedents (reseaux de neurones heuristiques, principes d'optimisation, codage predictif, ...) peuvent se rapporter a ce langage commun. %
#     #
#     # ... when including action in such models, it becomes ...
#     #
#     # """)
#
#     s.add_slide(content=s.content_figures(
#     [os.path.join(figpath, figname) for figname in ['Friston12.png', 'Adams12.png', 'PerrinetAdamsFriston14header_small.png']], bgcolor="white",
#     #title=title,
#     fragment=True,
#     transpose=True, height=s.meta['height']*height_ratio,
#     url=['http://invibe.net/LaurentPerrinet/Publications/' + name for name in ['Friston12', 'Adams12', 'PerrinetAdamsFriston14']]),
#     notes="""
# * in a first study, we have proposed that PERCEPTION (following Helmhotz 1866) is an active process of hypothesis testing by which we seek to confirm our predictive models of the (hidden) world: Active inference: (cite TITLE). In theory, one could find any prior to fit any experimental data, but the beauty of the theory comes from the simplicity of the models chosen to model the data at hand, such as saccades...
#
# * ... and even better if these models may find a possible correspondance into the neural anatomy and explain some deviation to a control  behaviour, such as that we modelled for understanding some aspects of the EMs of schizophrenic patients
#
# * Today, I will again focus on the problem of sensorimotor delays in the optimal control of (smooth) eye movements under uncertainty. Specifically, we consider delays in the visuo-oculomotor loop and their implications for active inference and I will present the results presented in the following paper (show TITLE).
#
# """)
#
#
#     # figpath = os.path.join(home, 'tmp/2015_RTC/2014-12-31_PerrinetAdamsFriston14/poster/12-06-25_AREADNE/')
#     freemove_bib = s.content_bib("LP, Adams and Friston", "2015", 'Biological Cybernetics, <a href="http://invibe.net/LaurentPerrinet/Publications/PerrinetAdamsFriston14">http://invibe.net/LaurentPerrinet/Publications/PerrinetAdamsFriston14</a>')
#
#     #for fname in ['figure1.png', 'figure2.png']:
#     figpath_law = os.path.join(home, 'quantic/2016_science/2016-10-13_LAW/figures')
#     figpath = 'figures/'
#     for figpath, fname, note in zip([figpath_law, figpath_law, 'figures/', 'figures/'], ['friston_figure1.png', 'friston_figure2.png', 'PAF14equations.png', 'PAF14equations2.png'], ["""
#
# * This schematic shows the dependencies among various quantities modelling exchanges of an agent with the environment. It shows the states of the environment and the system in terms of a probabilistic dependency graph, where connections denote directed (causal) dependencies. The quantities are described within the nodes of this graph -- with exemplar forms for their dependencies on other variables.
#
# * Hidden (external) and internal states of the agent are separated by action and sensory states. Both action and internal states -- encoding a conditional probability density function over hidden states -- minimise free energy. Note that hidden states in the real world and the form of their dynamics can be different from that assumed by the generative model; (this is why hidden states are in bold. )
# ""","""
# *  Active inference uses a generalisation of Kalman filtering to provide Bayes optimal estimates of hidden states and action in generalized coordinates of motion. As we have seen previously, the central nervous system has to contend with axonal delays, both at the sensory and the motor levels. Representing hidden states in generalized coordinates provides a simple way of compensating for both these delays.
#
# * This mathematical framework can be mapped to the anatomy of the visual system. Similar to the sketch that we have shown above, "compiling" (that is, solving) the equations of Free-energy minimization forms a set of coupled differential equations which correpond to different node along the visuo-oculomotor pathways.
# ""","""
#
# * a novelty of our approach including known delays was to take advantage of genralized coordinates to create an operator $T$ to travel back and forth in time with a delay $\tau$. It is simply formed by using a Taylor expansion of the succesive orders in the generalized coordinates which takes this form in matrix form and thus simply by taking the exponential matrix form.
# ""","""
# Applying such an operator to the FEM generates a slightly different and more complicated formulation but it is important to note that to compensate for delays, there is no change in the structure of the network but just in how the synaptic weights are tuned (similar to what we had done in the first part)
#
# * The efficacy of this scheme will be illustrated using neuronal simulations of pursuit initiation responses, with and without compensation.
#
# """]):
#         s.add_slide(#image_fname=os.path.join(figpath, fname),
#         content=s.content_figures(
#     [os.path.join(figpath, fname)], bgcolor="white",
#     #title=title,
#      height=s.meta['height']*height_ratio),# + freemove_bib,
# # >>> Lup IS HERE <<<
#     notes=note)
#
#     figpath = os.path.join(home, 'quantic/2016_science/2016-10-13_LAW/figures')
#
#
#     s.add_slide(content="""
#         <video controls autoplay loop width=99%/>
#           <source type="video/mp4" src="{}">
#         </video>
#         """.format(s.embed_video(os.path.join(figpath, 'flash_lag_dot.mp4'))),
#     notes="""
#
#     Pursuit initiation
#
#     """)
#     #figpath = os.path.join(home, 'tmp/2015_RTC/2014-12-31_PerrinetAdamsFriston14/poster/12-06-25_AREADNE/')
#     #for fname in ['Slide3.png', 'Slide4.png']:
#     #figpath = os.path.join(home, 'tmp/2015_RTC/2014-04-17_HDR/friston/')
#     for fname in ['friston_figure3.png',
#                   'friston_figure4-A.png',
#                   'friston_figure4-B.png',
#                    'friston_figure4-C.png']:
#
#         s.add_slide(#image_fname=os.path.join(figpath, fname),
#         content=s.content_figures(
#     [os.path.join(figpath, fname)], bgcolor="white",
#     #title=title,
#      height=s.meta['height']*height_ratio),# + freemove_bib,
#     notes="""
#
# This figure reports the conditional estimates of hidden states and causes during the simulation of pursuit initiation, using a single rightward (positive) sweep of a visual target, while compensating for sensory motor delays. We will use the format of this figure in subsequent figures: the upper left panel shows the predicted sensory input (coloured lines) and sensory prediction errors (dotted red lines) along with the true values (broken black lines). Here, we see horizontal excursions of oculomotor angle (upper lines) and the angular position of the target in an intrinsic frame of reference (lower lines). This is effectively the distance of the target from the centre of gaze and reports the spatial lag of the target that is being followed (solid red line). One can see clearly the initial displacement of the target that is suppressed after a few hundred milliseconds. The sensory predictions are based upon the conditional expectations of hidden oculomotor (blue line) and target (red line) angular displacements shown on the upper right. The grey regions correspond to 90% Bayesian confidence intervals and the broken lines show the true values of these hidden states. One can see the motion that elicits following responses and the oculomotor excursion that follows with a short delay of about 64ms. The hidden cause of these displacements is shown with its conditional expectation on the lower left. The true cause and action are shown on the lower right. The action (blue line) is responsible for oculomotor displacements and is driven by the proprioceptive prediction errors.
#
# * This figure illustrates the effects of sensorimotor delays on pursuit initiation (red lines) in relation to compensated (optimal) active inference -- as shown in the previous figure (blue lines). The left panels show the true (solid lines) and estimated sensory input (dotted lines), while action is shown in the right panels. Under pure sensory delays (top row), one can see clearly the delay in sensory predictions, in relation to the true inputs. The thicker (solid and dotted) red lines correspond respectively to (true and predicted) proprioceptive input, reflecting oculomotor displacement. The middle row shows the equivalent results with pure motor delays and the lower row presents the results with combined sensorimotor delays. Of note here is the failure of optimal control with oscillatory fluctuations in oculomotor trajectories, which become unstable under combined sensorimotor delays.
#
# """)
#
#     #figpath = 'figures/'
#     s.add_slide(content="""
#         <video controls autoplay loop width=99%/>
#           <source type="video/mp4" src="{}">
#         </video>
#         """.format(s.embed_video(os.path.join(figpath, 'flash_lag_sin.mp4'))),
#     notes="""
#
#     Smooth Pursuit
# We then consider an extension of the generative model to simulate smooth pursuit eye movements --- in which the visuo-oculomotor system believes both the target and its centre of gaze are attracted to a (hidden) point moving in the visual field.
#     """)
#     #figpath = os.path.join(home, 'tmp/2015_RTC/2014-04-17_HDR/friston/')
#     for fname in ['friston_figure6.png', 'friston_figure7.png']:
#         s.add_slide(#image_fname=os.path.join(figpath, fname),
#         content=s.content_figures(
#     [os.path.join(figpath, fname)], bgcolor="white",
#     #title=title,
#      height=s.meta['height']*height_ratio),# + freemove_bib,
#     notes="""
#
#
# * This figure uses the same format as the previous figure -- the only difference is that the target motion has been rectified so that it is (approximately) hemi-sinusoidal. The thing to note here is that the improved accuracy of the pursuit previously apparent at the onset of the second cycle of motion has now disappeared -- because active inference does not have access to the immediately preceding trajectory. This failure of an anticipatory improvement in tracking is contrary to empirical predictions. \item \odot
#
# * the generative model has been equipped with a second hierarchical level that contains hidden states, modelling latent periodic behaviour of the (hidden) causes of target motion. With this addition, the improvement in pursuit accuracy apparent at the onset of the second cycle of motion is reinstated. This is because the model has an internal representation of latent causes of target motion that can be called upon even when these causes are not expressed explicitly in the target trajectory.
#
# """)
#
#     s.add_slide(content="""
#         <video controls autoplay loop width=99%/>
#           <source type="video/mp4" src="{}">
#         </video>
#         """.format(s.embed_video(os.path.join(figpath, 'flash_lag_sin2.mp4'))),
#     notes="""
#
#     Smooth Pursuit with oclusion
# Finally, the generative model is equipped with a hierarchical structure, so that it can recognise and remember unseen (occluded) trajectories and emit anticipatory responses.
#     """)
#     #figpath = os.path.join(home, 'tmp/2015_RTC/2014-12-31_PerrinetAdamsFriston14/poster/12-06-25_AREADNE/')
#
#     #figpath = os.path.join(home, 'tmp/2015_RTC/2014-04-17_HDR/friston/')
#     for fname in ['friston_figure8.png', 'friston_figure9bis.png']:
#         s.add_slide(#image_fname=os.path.join(figpath, fname),
#         content=s.content_figures(
#     [os.path.join(figpath, fname)], bgcolor="white",
#     #title=title,
#      height=s.meta['height']*height_ratio),# + freemove_bib,
#     notes="""
#
#
# * This figure uses the same format as the previous figure -- the only difference is that the target motion has been rectified so that it is (approximately) hemi-sinusoidal. The thing to note here is that the improved accuracy of the pursuit previously apparent at the onset of the second cycle of motion has now disappeared -- because active inference does not have access to the immediately preceding trajectory. This failure of an anticipatory improvement in tracking is contrary to empirical predictions. \item \odot
#
# * the generative model has been equipped with a second hierarchical level that contains hidden states, modelling latent periodic behaviour of the (hidden) causes of target motion. With this addition, the improvement in pursuit accuracy apparent at the onset of the second cycle of motion is reinstated. This is because the model has an internal representation of latent causes of target motion that can be called upon even when these causes are not expressed explicitly in the target trajectory.
#
# """)
#
#     s.close_section()
#


i_section += 1
if do_section[i_section]:
    s.open_section()
    title = meta['sections'][i_section]
    s.add_slide_outline(i_section)

    s.add_slide(content=intro,
    notes="""
Thanks for you attention!
""")
    s.close_section()


s.compile(filename=slides_filename)
