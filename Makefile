default: show

SRC=2018-03-26_cours-NeuroComp_FEP

edit:
	atom $(SRC).py

html:
	python3 $(SRC).py $(SRC).html

page:
	open https://invibe.net/cgi-bin/index.cgi/Presentations/$(SRC)?action=edit

show: html
	open -a safari $(SRC).html

blog:
	python3 $(SRC).py $(SRC).html
	rsync -av src/2017-01-15-bogacz-2017-a-tutorial-on-free-energy.ipynb ~/pool/blog/invibe/posts/
	rsync -av src/2017-03-09_probabilities.ipynb ~/pool/blog/invibe/posts/
	rsync -av $(SRC).html ~/pool/blog/invibe/files/
	cd ~/pool/blog/invibe/ ; nikola build
	cd ~/pool/blog/invibe/ ; nikola deploy
	open -a safari http://blog.invibe.net/files/$(SRC).html
