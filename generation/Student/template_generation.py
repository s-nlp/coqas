import json
import random
import inflect

import spacy 
nlp = spacy.load("en_core_web_sm") 

def create_aspect_list(aspect_list, another_list = []):
    resulted_list = []
    aspect_set = set(aspect_list)
    for elem in aspect_set:
        doc = nlp(elem)
        if (len(elem.split()) > 1 and elem not in another_list and len(resulted_list) < 3):
            resulted_list.append(elem)
        elif ((doc[0].tag_ == 'JJR' or doc[0].tag_ == 'RBR') and elem not in another_list and len(resulted_list) < 3):
            resulted_list.append(elem)
    return resulted_list


def generate_template(comparing_pair, mode='default'):
    if mode == "extended" and len(comparing_pair['winner_aspects']) > 0 and len(comparing_pair['loser_aspects']) > 0:
        print ("generate template 0")
        extented_templates = []
        first_comparing_sentence_parts = []
        second_comparing_sentence_parts = []
        first_comparing_sentence_parts.append("I would prefer to use {winner} because of {win_asp}.")
        first_comparing_sentence_parts.append("Looks like {winner} is better, because of {win_asp}.")
        first_comparing_sentence_parts.append("It's simple! {Winner} is better, because of {win_asp}.")
        first_comparing_sentence_parts.append("After much thought, I realized that  {winner} is better, because of {win_asp}.")
        first_comparing_sentence_parts.append("I came to the conclusion that {winner} is better, because of {win_asp}.")
        second_comparing_sentence_parts.append(" {Looser} is {lose_asp}.")
        second_comparing_sentence_parts.append(" But you should know that {looser} is {lose_asp}.")
        second_comparing_sentence_parts.append(" But it will be useful for you to know that {looser} is {lose_asp}.")
        second_comparing_sentence_parts.append(" But i should tell you that {looser} is {lose_asp}.")
        print ("generate template 1")
        for i in range(len(first_comparing_sentence_parts)):
            for j in range(len(second_comparing_sentence_parts)):
                extented_templates.append(first_comparing_sentence_parts[i] + second_comparing_sentence_parts[j])

        template_index = random.randint(0, len(extented_templates) - 1)
        ordinal = True #bool(random.getrandbits(1))
        print ("generate template 22")
        print (", ".join(comparing_pair['winner_aspects']))
        print (", ".join(comparing_pair['winner_aspects']))
        if (len(comparing_pair['winner_aspects']) < 1):
            winner_aspects_string = "looks better"
        elif (len(comparing_pair['winner_aspects']) == 1):
            winner_list = comparing_pair['winner_aspects']
            winner_aspects_string = comparing_pair['winner_aspects'][0]
        else:
            winner_list = create_aspect_list(comparing_pair['winner_aspects'])
            winner_aspects_string = ", ".join(winner_list[:-1]) + ' and ' + winner_list[-1]
        print ("winner_aspects_string1", winner_aspects_string)
        if (len(comparing_pair['loser_aspects']) < 1):
            loser_aspects_string = "not promising"
        elif (len(comparing_pair['loser_aspects']) == 1):
            loser_aspects_string = comparing_pair['loser_aspects'][0]
        else:
            loser_list = create_aspect_list(comparing_pair['loser_aspects'], winner_list)
            loser_aspects_string = ", ".join(loser_list[:-1]) + ' and ' + loser_list[-1]
        print ("looser_aspects_string1", loser_aspects_string)
        response = extented_templates[template_index].format(winner = comparing_pair['winner'],
                                                             win_asp = winner_aspects_string,
                                                             looser = comparing_pair['loser'], 
                                                             lose_asp = loser_aspects_string, 
                                                             Looser = comparing_pair['loser'].capitalize(),
                                                             Winner = comparing_pair['winner'].capitalize())
        print ("make response ", response)
    else:
        mode = "default"
    if mode == 'default':
        print ("generate template default")
        if len(comparing_pair['winner_aspects']) > 0:
            response = "It seems like {} is better than {} because it is {}.".format(comparing_pair['winner'],
                                                                                      comparing_pair['loser'],
                                                                                      ", ".join(comparing_pair[
                                                                                                    'winner_aspects']))
        elif len(comparing_pair['loser_aspects']) > 0:
            response = "Looks like {} is better than {}, but {} is {}.".format(comparing_pair['winner'],
                                                                                comparing_pair['loser'],
                                                                                comparing_pair['loser'], ", ".join(
                    comparing_pair['loser_aspects']))
        else:
            response = "I would prefer {} than {}.".format(comparing_pair['winner'], comparing_pair['loser'])
    return response


# data = []
# with open('mined_bow_str.json') as f:
#     for line in f:
#         data.append(json.loads(line))
# answer_list = []
# for line in data:
#     comparing_pair = {}
#     comparing_pair['winner'] = line['winner']
#     if line['object1']['name'] == line['winner']:
#         winner_tag = "Object1"
#         loser_tag = 'Object2'
#         comparing_pair['loser'] = line['object2']['name']
#     else:
#         winner_tag = "Object2"
#         loser_tag = "Object1"
#         comparing_pair['loser'] = line['object1']['name']
#
#     comparing_pair['winner_aspects'] = line['extractedAspects' + winner_tag]
#     comparing_pair['loser_aspects'] = line['extractedAspects' + loser_tag]
#     template = get_template(comparing_pair, mode='extended')
#     print(template)
