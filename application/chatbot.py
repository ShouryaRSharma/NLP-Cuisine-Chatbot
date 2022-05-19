import random
import logging 

from utility_methods import load_model
from words import GREETING, ORDERING, REJECTIONS, CONFIRMATIONS, CANCELS, UPDATES, FOLLOWUPS, ORDER, STUPID, FARWELLS, APPROVAL, DISHES

# TextPreparationVectorizer, NormalizationScaler are needed for pickling of the cuisine_classifier
from ic_components import CUISINES_STR, SUB_TAG_CLASSIFIER, TextPreparationVectorizer, NormalizationScaler
from ner_components import ENTITIES


class ChatBot:

    def __init__(self, name, path=r"./application", log_level=logging.WARNING):

        self.name = name

        self.response  = []
        self.utterance = None

        self.__inital_conversation_state()

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


    def __inital_conversation_state(self):

        self.in_conversation = False

        self.awaiting_order_confirmation = False 
        self.awaiting_remove_confirmation = False
        self.awaiting_rejection = False
   
        self.cuisine = None
        self.dishes = []

        self.intents = []
        self.entities = {}

        self.attempts = 0
        self.suggestions = 0        

    def intent_classification(self):

        intents = SUB_TAG_CLASSIFIER.predict([self.utterance])[0]
        self.intents = intents

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

        # Remove previous suggestion if it has not been confirmed.
        if self.suggestions > 0:
            self.dishes[:-1]
        self.dishes.append(random.choice(DISHES[self.cuisine]))

        self.response.append(f"Can I recommend you to {random.choice(ORDER)} {self.__order_query()}?")
        self.awaiting_order_confirmation = True

        self.suggestions += 1


    def order(self):

        # If entities does not include any DISHES, apologies or suggest and ask again.
        if not any(self.entities["DISH"]):

            if self.attempts < 1:
                self.response.append(f"{random.choice(ORDERING)}")

            if 1 <= self.attempts < 2:
                self.suggest_order()

            if self.attempts >= 2:
                self.response.append(random.choice(STUPID))

            self.attempts += 1


        # If entities does include any DISHES, add them and ask for confirmation.
        if any(self.entities["DISH"]):  

            self.cuisine_classification()
            self.dishes.extend(self.entities["DISH"])

            self.response.append(f"Would you like to {random.choice(ORDER)} {self.__order_query()}?")

            self.attempts    = 0
            self.suggestions = 0
            self.awaiting_order_confirmation = True


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


    def __awaiting_order_confirmation(self):

        if all(i not in ["confirm", "reject"] for i in self.intents): 
            self.response.append(f"{random.choice(APPROVAL)} that you would like to {random.choice(ORDER)} {self.__order_query(cuisine=False)}.")

        if "confirm" in self.intents: 

            self.response.append(f"{random.choice(CONFIRMATIONS)} your order of {self.__order_query(cuisine=False)}.")
            self.awaiting_order_confirmation = False
            self.attempts    = 0
            self.suggestions = 0

            self.response.append(f"{random.choice(FOLLOWUPS)}")
            self.awaiting_rejection = True

        if "reject" in self.intents:

            # If no, start over
            self.cuisine = None
            self.dishes = []

            self.response.append(f"{random.choice(REJECTIONS)}. Please repeat your order.")
            self.awaiting_order_confirmation = False


    def __awaiting_rejection(self):

        if all(i not in ["confirm", "reject"] for i in self.intents): 
            self.response.append(f"{random.choice(APPROVAL)} if you need or don't need anything else.")

        if "reject" in self.intents:

            # If nothing else to do, finalize conversation
            self.response.append(f"Your order of {self.__order_query(cuisine=False)} is on your way.")
            self.__finalize_conversation()

        if "confirm" in self.intents: 

            # If yes, check other possible intents
            self.awaiting_rejection = False


    def remove_order(self):
 
        # Await confirmation if it is a complete cancel or update only
        self.response.append(f"Do you want to cancel your order completely or just update {self.__order_query(cuisine=False)}?")
        self.awaiting_remove_confirmation = True


    def __awaiting_remove_confirmation(self):

        if all(i not in ["update", "cancel"] for i in self.intents): 
            self.response.append(f"{random.choice(APPROVAL)} if you want to cancel your order completely or just update it.")

        if "cancel" in self.intents: 

            self.response.append(f"{random.choice(CANCELS)} of {self.__order_query(cuisine=False)}.")
            self.awaiting_remove_confirmation = False

            # If cancel, end conversation
            self.__finalize_conversation()

        if "update" in self.intents:

            # If update, start over
            self.cuisine = None
            self.dishes = []

            self.response.append(f"{random.choice(UPDATES)}")
            self.awaiting_remove_confirmation = False


    def default_answer(self):

        if  self.attempts < 1:
            self.response.append(f"{random.choice(FOLLOWUPS)}")
            self.awaiting_rejection = True

        if 1 <= self.attempts < 3:
            self.suggest_order()
        
        if self.attempts >= 3:
            self.__finalize_conversation()
        
        self.attempts += 1


    def dialog_flow_manager(self):

        if not self.in_conversation:
            self.__initialize_conversation()

        else:
            if self.awaiting_order_confirmation:
                self.__awaiting_order_confirmation()

            if self.awaiting_rejection: 
                self.__awaiting_rejection()

            if self.awaiting_remove_confirmation:
                self.__awaiting_remove_confirmation()


        if (not any([self.awaiting_order_confirmation, 
                    self.awaiting_rejection, 
                    self.awaiting_remove_confirmation]) 
                    and self.in_conversation 
                    and not self.response):

            if "order" in self.intents:
                self.order()

            if any(i in ["update", "cancel"] for i in self.intents):
                self.remove_order()

            if not self.intents:
                self.suggest_order()

        if "farewell" in self.intents:
            self.__finalize_conversation()
        
        if not self.response:
            self.default_answer()

        log_msg = [f"Model Variables:\n",
                   f"\t\t\tIn Conversation: {self.in_conversation}\n",
                   f"\t\t\tDishes: {self.dishes}\n"
                   f"\t\t\tAwaiting order confirmation: {self.awaiting_order_confirmation}\n",
                   f"\t\t\tAwaiting remove confirmation: {self.awaiting_remove_confirmation}\n",
                   f"\t\t\tAwaiting rejection: {self.awaiting_rejection}\n",
                   f"\t\t\tNumber of attempts: {self.attempts}\n",
                   f"\t\t\tNumber of suggestions: {self.suggestions}\n",]

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
    