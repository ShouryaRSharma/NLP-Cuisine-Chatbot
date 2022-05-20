import random
import logging 

from utility_methods import load_model
from words import GREETING, ORDERING, REJECTIONS, CONFIRMATIONS, CANCELS, UPDATES, FOLLOWUPS, ORDER, STUPID, FARWELLS, APPROVAL, DISHES

# TextPreparationVectorizer, NormalizationScaler are needed for pickling of the cuisine_classifier
from ic_components import CUISINES_STR, SUB_TAG_CLASSIFIER, TextPreparationVectorizer, NormalizationScaler
from ner_components import ENTITIES


class ChatBot:

    def __init__(self, name, path=r"./application", log_level=logging.WARNING, intent_ai=False):

        self.name = name
        self.intent_ai = intent_ai

        self.response  = []
        self.utterance = None

        self.__inital_conversation_state()

        self.intent_classifier        = load_model("intent_classifier.pickle",        path=f"{path}/models/")
        self.cuisine_classifier       = load_model("cuisine_classifier.pickle",       path=f"{path}/models/")
        self.named_entity_recognitior = load_model("named_entity_recognitior.pickle", path=f"{path}/models/")

        # Set logging levels either to: INFO, DEBUG (default WARNING will ignore those two)
        # Restart of kernel is needed for it to pick up
        logging.basicConfig(level=log_level)

        self.conv_logger = logging.getLogger("conversation")
        conv_handler = logging.FileHandler(f"{path}/logging/conversation.log", mode="w")  

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        conv_handler.setFormatter(formatter)

        self.conv_logger.addHandler(conv_handler)


    def __clear_order(self): 

        self.cuisine = None
        self.dishes = []


    def __clear_attempts(self, clear_awating=False):

        self.attempts = 0
        self.suggested = False  

        if clear_awating:
            self.awaiting = None


    def __inital_conversation_state(self):

        self.in_conversation = False

        # order_confirmation, remove_confirmation, reject_confirmation
        self.__clear_attempts(clear_awating=True)
        self.__clear_order() 

        self.intents = []
        self.entities = {}

   
    def intent_classification(self):

        if self.intent_ai:

            labels = ['greeting','order', 'update', 'farewell' , 'cancel', 'confirm', 'reject']

            pred, _ = self.intent_classifier.predict([self.utterance])
            predArr = []
            for i in range(len(pred[0])):
                if pred[0][i] == 1:
                    predArr.append(labels[i])
                    
            self.conv_logger.debug(f"AI Predicted Intents: {predArr}")

        self.intents = SUB_TAG_CLASSIFIER.predict([self.utterance])[0]
        
        self.conv_logger.debug(f"Predicted Intents: {self.intents}")
    

    def cuisine_classification(self):

        cuisine = None

        # Check if an explicit cuisine is mentioned
        for name in CUISINES_STR:
            cus = name.lower()
            if cus in self.utterance.lower():
                cuisine = cus.capitalize()
        
        # If not mentioned, use classifier to detect intent 
        if not cuisine:
            cuisine = CUISINES_STR[self.cuisine_classifier.predict([self.utterance])[0]] 

        self.cuisine = cuisine

        self.conv_logger.debug(f"Predicted Cuisine: {self.cuisine}")


    def named_entity_recognition(self):
        
        # Create dictionary for the entities
        ner_dict = {entity: [] for entity in ENTITIES}
        text_prediction, _ = self.named_entity_recognitior.predict([self.utterance])
        
        for item in text_prediction[0]:
            
            key   = list(item.values())[0]
            value = list(item.keys())[0]

            # Ignore non-named entity
            if key != "O":
                ner_dict[key[2:]].append(value)
        
        self.entities = ner_dict

        self.conv_logger.debug(f"Predicted NER: {self.entities}")


    def __order_query(self, cuisine=True):

        dishes = ", ".join(self.dishes)
        cuisine_str = ""

        if cuisine:
            cuisine_str = f", from the {self.cuisine} cuisine"

        return f"{dishes}{cuisine_str}"


    def suggest_order(self):

        self.cuisine_classification()
        self.dishes.append(random.choice(DISHES[self.cuisine]))

        self.response.append(f"Can I recommend you to {random.choice(ORDER)} {self.__order_query()}?")

        self.awaiting = "order_confirmation"
        self.suggested = True


    def order(self):

        # If entities does not include any DISHES, apologies or suggest and ask again.
        if not any(self.entities["DISH"]):

            if self.attempts < 1:
                self.response.append(f"{random.choice(ORDERING)}")

            self.__attempt_answers()

        # If entities does include any DISHES, add them and ask for confirmation.
        if any(self.entities["DISH"]):  

            self.cuisine_classification()
            self.dishes.extend(self.entities["DISH"])

            self.response.append(f"Would you like to {random.choice(ORDER)} {self.__order_query()}?")

            self.__clear_attempts(clear_awating=True)
            self.awaiting = "order_confirmation"


    def remove_order(self):
 
        # Await confirmation if it is a complete cancel or update only
        self.response.append(f"Do you want to cancel your order completely or just update {self.__order_query(cuisine=False)}?")

        self.__clear_attempts(clear_awating=True)
        self.awaiting = "remove_confirmation"


    def __initialize_conversation(self):

        self.in_conversation = True
        greeting = random.choice(GREETING).capitalize()
        self.response.append(f"{greeting}!")

        # Check if it already has asked for order and if not, ask for order.
        self.order()


    def __finalize_conversation(self):

        self.response.append(f"{random.choice(FARWELLS)}")

        # Clear all variables for next conversation.
        self.__inital_conversation_state()


    def __awaiting_rejection(self):

        if all(i not in ["confirm", "reject"] for i in self.intents): 
            if self.attempts < 2:
                self.response.append(f"{random.choice(APPROVAL)} if you need or don't need anything else.")
                self.attempts += 1
            else:
                self.awaiting = None
                self.__attempt_answers()

        if "reject" in self.intents:

            # If nothing else to do, finalize conversation
            if self.dishes:
                self.response.append(f"Your order of {self.__order_query(cuisine=False)} is on your way.")
            self.__finalize_conversation()

        if "confirm" in self.intents: 

            # If yes, check other possible intents
            self.awaiting = None
            self.__order_intent()


    def __awaiting_order_confirmation(self):

        if all(i not in ["confirm", "reject"] for i in self.intents):
            if self.attempts < 2:
                self.response.append(f"{random.choice(APPROVAL)} that you would like to {random.choice(ORDER)} {self.__order_query(cuisine=False)}.")
                self.attempts += 1
            else:
                self.awaiting = None
                self.__attempt_answers()

        if "confirm" in self.intents: 

            self.__clear_attempts(clear_awating=True)

            self.response.append(f"{random.choice(CONFIRMATIONS)} your order of {self.__order_query(cuisine=False)}.")
            self.__anything_else()

        if "reject" in self.intents:

            self.__clear_attempts(clear_awating=True)

            if self.suggested:
                self.dishes[:-1]
                self.response.append(f"{random.choice(REJECTIONS)}")
                self.__anything_else()
            else:
                self.__clear_order() 
                self.response.append(f"{random.choice(REJECTIONS)}. Please repeat your order.")


    def __awaiting_remove_confirmation(self):

        if all(i not in ["update", "cancel"] for i in self.intents): 
            if self.attempts < 2:
                self.response.append(f"{random.choice(APPROVAL)} if you want to cancel your order completely or just update it.")
                self.attempts += 1
            else:
                self.awaiting = None
                self.__attempt_answers()

        if "cancel" in self.intents: 

            self.response.append(f"{random.choice(CANCELS)} of {self.__order_query(cuisine=False)}.")
            
            # If cancel, clear and end conversation
            self.__finalize_conversation()

        if "update" in self.intents:

            # If update, start over
            self.__clear_order() 

            self.response.append(f"{random.choice(UPDATES)}")
            self.__clear_attempts(clear_awating=True)


    def default_answer(self):

        if  self.attempts < 1:
            self.__anything_else()

        self.__attempt_answers()


    def __attempt_answers(self):

        if 0 < self.attempts < 2:
            self.suggest_order()
        
        if 1 < self.attempts < 3:
            self.response.append(random.choice(STUPID))

        if 2 < self.attempts < 4:
            self.__finalize_conversation()
        
        if self.in_conversation:
            self.attempts += 1


    def __anything_else(self):

        self.response.append(f"{random.choice(FOLLOWUPS)}")
        self.awaiting = "reject_confirmation"

    
    def __order_intent(self):

        if "order" in self.intents:
            self.order()

        if any(i in ["update", "cancel"] for i in self.intents):
            self.remove_order()


    def dialog_flow_manager(self):

        if not self.in_conversation:
            self.__initialize_conversation()
        else:
            if self.awaiting:

                if self.awaiting == "reject_confirmation": 
                    self.__awaiting_rejection()

                elif self.awaiting == "order_confirmation":
                    self.__awaiting_order_confirmation()

                elif self.awaiting == "remove_confirmation":
                    self.__awaiting_remove_confirmation()
            else:
                self.__order_intent()

        if self.in_conversation and not self.response:
            self.default_answer()

        log_msg = [f"Model Variables:\n",
                   f"\t\t\tIn Conversation: {self.in_conversation}\n",
                   f"\t\t\tDishes: {self.dishes}\n"
                   f"\t\t\tAwaiting: {self.awaiting}\n",
                   f"\t\t\tNumber of attempts: {self.attempts}\n",
                   f"\t\t\tSuggestions: {self.suggested}\n",]

        self.conv_logger.debug("".join(log_msg))


    def ask(self, utterance):

        self.conv_logger.info(f"USER: {utterance}\n")

        # Main entrance method for receiving a response
        self.utterance = utterance
        self.response = []

        self.intent_classification()
        self.named_entity_recognition()
        self.dialog_flow_manager()

        response = " ".join(self.response)

        self.conv_logger.info(f"CHATBOT: {response}\n\n\n")

        return response


    def debug(self, utterance, intents=None, entities=None):

        self.utterance = utterance
        self.response = []

        # Manually select intents of the query
        if not intents:
            self.intent_classification()
        else:
            self.intents = intents

        # Manually select the entities of the query
        if not entities:
            self.named_entity_recognition()
        else:
            self.entities = entities

        self.dialog_flow_manager()
        
        return " ".join(self.response)
    