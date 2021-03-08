import json
import random
import inflect


def generate_template(comparing_pair, mode='default'):
    if mode == "extended" and len(comparing_pair['winner_aspects']) > 0 and len(comparing_pair['loser_aspects']) > 0:
        extented_templates = []
        first_comparing_sentence_parts = []
        second_comparing_sentence_parts = []
        first_comparing_sentence_parts.append("I would prefer to use {} because it is: {}")
        first_comparing_sentence_parts.append("Looks like {} is better, because: {}")
        first_comparing_sentence_parts.append("It's simple! {} is better, because: {}")
        first_comparing_sentence_parts.append("after much thought, I realized that  {} is better, because: {}")
        first_comparing_sentence_parts.append("i came to the conclusion that {} is better, because: {}")
        second_comparing_sentence_parts.append(", but {} is: {}")
        second_comparing_sentence_parts.append(". But you should know that {} is: {}")
        second_comparing_sentence_parts.append(". But it will be useful for you to know that {} is: {}")
        second_comparing_sentence_parts.append(". But i should tell you that {} is: {}")
        for i in range(len(first_comparing_sentence_parts)):
            for j in range(len(second_comparing_sentence_parts)):
                extented_templates.append(first_comparing_sentence_parts[i] + second_comparing_sentence_parts[j])

        template_index = random.randint(0, len(extented_templates) - 1)
        ordinal = bool(random.getrandbits(1))
        if ordinal and (len(comparing_pair['winner_aspects']) < 4 or len(comparing_pair['loser_aspects']) < 4):
            winner_aspects_string = ""
            loser_aspects_string = ""
            p = inflect.engine()
            if len(comparing_pair['winner_aspects']) > 1:
                for i in range(len(comparing_pair['winner_aspects']) - 1):
                    winner_aspects_string += str(p.number_to_words(p.ordinal(i + 1))) + ", " + \
                                             comparing_pair['winner_aspects'][i] + ", "
                winner_aspects_string += str(p.number_to_words(p.ordinal(len(comparing_pair['winner_aspects'])))) + ", " + \
                                         comparing_pair['winner_aspects'][-1]
            else:
                winner_aspects_string = comparing_pair['winner_aspects'][0]
            if len(comparing_pair['loser_aspects']) > 1:
                for i in range(len(comparing_pair['loser_aspects']) - 1):
                    loser_aspects_string += str(p.number_to_words(p.ordinal(i + 1))) + ", " + \
                                            comparing_pair['loser_aspects'][i] + ", "
                loser_aspects_string += str(p.number_to_words(p.ordinal(len(comparing_pair['loser_aspects'])))) + ", " + \
                                        comparing_pair['loser_aspects'][-1]
            else:
                loser_aspects_string = comparing_pair['loser_aspects'][0]
            response = extented_templates[template_index].format(comparing_pair['winner'], winner_aspects_string,
                                                                 comparing_pair['loser'], loser_aspects_string)
        else:
            response = extented_templates[template_index].format(comparing_pair['winner'],
                                                                 ", ".join(comparing_pair['winner_aspects']),
                                                                 comparing_pair['loser'],
                                                                 ", ".join(comparing_pair['loser_aspects']))
    else:
        mode = "default"
    if mode == 'default':
        if len(comparing_pair['winner_aspects']) > 0:
            response = "It seems like {} is better than {} because it is: {}.".format(comparing_pair['winner'],
                                                                                      comparing_pair['loser'],
                                                                                      ", ".join(comparing_pair[
                                                                                                    'winner_aspects']))
        elif len(comparing_pair['loser_aspects']) > 0:
            response = "Looks like {} is better than {}, but {} is: {}.".format(comparing_pair['winner'],
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
