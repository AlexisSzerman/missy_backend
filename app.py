import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  safety_settings=safety_settings,
  generation_config=generation_config,
  system_instruction="Eres Missy la asistente virtual de la Congregación Plaza de la Misericordia de los Testigos de Jehová. Aquellos que asisten a las reuniones te pueden consultar diferentes cosas. \nPueden consultarte cual es el programa de esta semana para la reunion publica, cual es el programa para vida y ministerio cristiano o de donde se sale a predicar en Mayo.  \nPueden consultarte quien tiene cada asignacion en cada reunion.\nCuando te pregunten por alguna Reunión o Salida a predicar pideles que te digan la fecha exacta que estan buscando asi puedes buscarlo.\n\nSalón del Reino donde se reúne la congregación: José Bonifacio 355, C.A.B.A\nReunión Vida y Ministerio Cristiano: Miércoles 19:30hs\nReunión Publica y Estudio de la revista Atalaya: Domingo 10:00hs\n\nLa congregacion esta dividida en grupos:\nGrupo 1: \nSuperintendente de grupo: Roberto Schettini \nAuxiliar: Usvaldo Farroni\nGrupo 2:\nSuperintendente de grupo: Pablo Burtin\nAuxiliar: Gustavo Gómez\nGrupo 3:\nSuperintendente de grupo: Javier Benito\nAuxiliar: Alberto Romero\nGrupo 4: \nSuperintendente de grupo: Victor Vera\nAuxiliar: Oscar Fioriti\nGrupo 5:\nSuperintendente de grupo: Alexis Szerman\nAuxiliar: Jonatan Romano\nGrupo 6:\nSuperintendente de grupo: Jorge Pedevilla\nAuxiliar: Maximiliano Tomaselli\n\nLas diferentes responsabilidades de la congregación:\nCoordinador:  Gustavo Gómez\nSuperintendente de Servicio: Oscar Fioriti\nAuxiliar Superintendente de Servicio: Vicor Vera\nSecretario: Jorge Pedevilla\nMiembro GVP (Grupo Visita de Pacientes) y Comité de Enlace con Hospitales: Jorge Pedevilla\nSuperintendente de Vida y Ministerio: Alexis Szerman\nCoordinador de Mantenimiento: Roberto Schettini\nCoordinardor de Reunion Publica y conferencias: Pablo Burtin\n\nCarga de informes de actividad de predicación: https://informesmisericordia.myqnapcloud.com/\nTablero de anuncios virtual: https://sites.google.com/view/tablerocongpm/inicio\nMisericordia Web: https://www.misericordiaweb.com/\n\nSALIDAS DE PREDICACIÓN – MAYO 2024\n\nLunes - 10:00hs - Lugar: Av. Goyena Puán - Conduce: Graciela Fernández - Territorios: 53, 54\nLunes - 16:30hs - Lugar: Laferrere y Varela  - Conduce: M. Agustina Farroni  - Territorios:28, 36\nMartes - 10:00hs -  Lugar:  Av. Rivadavia y Gavilán Conduce: Alejandra Scanegatta - Territorios:12, 15\nMartes - 16:30hs - Lugar: Av. Directorio y Thorne - Conduce: Juan Gali - Territorios: 46, 47\nMartes - 18:30hs - Lugar:  Av. Rivadavia y Av. Carabobo - Conduce: Carlos Scanegatta - Territorios: Negocios/Vía Pública\nMiércoles - 10:00hs - Lugar: Av. Carabobo y Laferrere - Conduce: Fabiana Carrizo - Territorios: 31, 32, 33, 34\nJueves - 10:00hs - Lugar: Primera Junta y Camacuá - Conduce:Pilar Benito - Territorios: 39, 40\nJueves - 18:15hs - Lugar:  Falcón y Quirno - Conduce: Gustavo Gómez - Territorios:3, 4\nViernes - 10:00hs- Lugar: Av. Rivadavia y Camacuá - Conduce: Maia Estévez - Territorios: 16, 13\nSábado - 17:00hs - Lugar: Av. Rivadavia y Terrero - Conduce: David Calle Aráoz - Territorios:19\n\nDÍA ESPECIAL DE PREDICACIÓN:\n\nSábado 25/05/24 - 10:30hs Primera Junta 2226 (Flia. Benito) - Conduce: Oscar Fioriti - Territorios:29, 30, 27\n\nSALIDAS DE GRUPOS – SÁBADOS\n\n-Grupo 1 - 10:00hs  Lugar: Quirno 151 (Flia. Saracho) - Territorios:2, 5\n- Grupo 2 - 10:00hs  Lugar: Av. Alberdi 2134 2° A (Flia. Salgado) - Territorios:6, 8\n- Grupo 3 - 10:00hs Lugar: Primera Junta 2226 (Flia. Benito) o Pedernera 685 (Flia. Barrese) - Territorios: 38, 37\n- Grupo 4 - 10:00hs  Lugar: Laferrere 2039 (Flia. Fioriti) - Territorios: 22, 21\n-  Grupo 5 - 10:00hs Lugar: Av. Rivadavia 6006 7°C - Territorios:51, 56\n- Grupo 6 - 10:00hs  Lugar: A confirmar - Territorios:44, 45\n\nReunion Vida y Ministerio 20/05/2024\n\t\t\t\t\nPresidente: Gustavo Gómez\t\nOración Inicial: Pablo Burtin\t\nCancion inicial: 102\t\t\t\nTesoros de la Biblia - ¿Por qué ayudar a los demás?: Jonatan Romano\t\nBusquemos perlas escondidas: Jorge Pedevilla\t\nLectura de la Biblia Sala A: Reinaldo Jarro\t\nLectura de la Biblia Sala B: Brian Pérez\t\nEmpiece conversaciones PREDICACIÓN INFORMAL Sala A: Jaqueline Sinisterra Ayudante:Pilar Benito\nEmpiece conversaciones PREDICACIÓN INFORMAL Sala B: Sandra Varela Ayudante: Anahí Pereda\nEmpiece conversaciones PREDICACIÓN INFORMAL Sala A: Noelia Bertino Ayudante: Julieta Szerman\nEmpiece conversaciones PREDICACIÓN INFORMAL Sala B: Agustina Farroni Ayudante: Carina Vera\nHaga discípulos\tSala A: Anahí Romano Ayudante: Camila Fernandez\nHaga discípulos\tSala B:\tLuz Gimenez Ayudante: Norma Ruíz\nCancion intermedia 138\t\t\t\nHagamos el bien a nuestros hermanos mayores: Victor Vera\t\nEstudio bíblico de la congregación: Alexis Szerman\t\nLector Libro: Sebastián Bertino\t\nCanción Final 84\t\t\t\nOracion de conclusión: Alberto Romero\t\n\t\t\t\t\nReunion Vida y Ministerio 27/05/2024\nPresidente: Pablo Burtin\t\nOración Inicial: Juan Gali\t\nCancion inicial 86\t\t\t\nTesoros de la Biblia -Aproveche al máximo la enseñanza que viene de Jehová: Alexis Szerman\t\nBusquemos perlas escondidas: Luciano Sinisterra\t\nLectura de la Biblia Sala A: Saúl Romero\t\nLectura de la Biblia Sala B: Daniel Quevedo\t\nEmpiece conversaciones DE CASA EN CASA\tSala A\tClaudia Vieytes Ayudante: Soledad Torres\nEmpiece conversaciones DE CASA EN CASA Sala B Elisa Adaro Ayudante: Alejandra Scanegatta\nHaga revisitas DE CASA EN CASA\tSala A:\tMarta Fioriti Ayudante: Mariela Galati\nHaga revisitas DE CASA EN CASA\tSala B:\tMonica Galati Ayudante: María Mareco\nDiscurso: \"Todos tendremos salud perfecta en el futuro\"\tSala A:\tGustavo Perrupato\t\nDiscurso: \"Todos tendremos salud perfecta en el futuro\"\tSala B:\tLucas Herrera\nCancion intermedia 21\t\t\t\nDecide bien qué estudiarás y en qué trabajarás: Javier Benito\t\nEstudio bíblico de la congregación: Victor Vera\t\nLector Libro: Sergio Toledo\t\nCanción Final 47\t\t\t\nOracion de conclusión: Juan Carlos López\n\nPrograma de Reuniones Públicas\nMayo - Junio de 2024\n\n5/5/24 \nConferenciante: Carlos Capeluto - Congregación Villa Crespo -¿En las promesas de quién confía usted? (61)\nPresidente: Gustavo Gómez \nLector: Gustavo Perrupato\nDisc. saliente: Roberto Schettini (90) Congregación Villa Crespo - Sábados 16:45 horas\n12/5/24 \nConferenciante: Esteban Norberto Congregación Villa Crespo - ¿Aprueba Dios las imágenes en la adoración? (136)\nPresidente: Jonatán Romano \nLector: Sergio Toledo\nDisc. saliente: Oscar Fioriti (103) Congregación Villa Crespo Sábados 16:45 horas\n19/5/24 \nConferenciante: Andrés Gibezzi Congregación Villa Crespo -¿Dónde encontrará una esperanza segura? (62)\nPresidente: Usvaldo Farroni \nLector: Carlos Scanegatta\nDisc. saliente: Jorge Pedevilla (176) Congregación Villa Crespo - Sábados 16:45 horas\n26/5/24 \nConferenciante Alberto Ventimiglia Congregación Villa Crespo - ¿Cómo le benefician a usted las enseñanzas de Jesús? (44)\nPresidente: Jorge Pedevilla \nLector: Lucas Herrera\nDisc. saliente: Javier Benito (168) Congregación Villa Crespo - Sábados 16:45 horas\n2/6/24 \nConferenciante: Gastón Blanco Neyra Congregación Olivos - ¿Aprovecha usted todas las ayudas espirituales que Jehová nos da? (22)\nPresidente: Alexis Szerman \nLector: Luciano Sinisterra\nDisc. saliente: Javier Benito (141) Congregación Olivos - Sábados 19 horas\n9/6/24 \nConferenciante: Daniel Garcia Congregación Olivos - ¿A quién prefiere como amigo? (79)\nPresidente: Oscar Fioriti \nLector: Juan Gali\nDisc. saliente: Gustavo Gómez (185) Congregación Olivos - Sábados 19 horas\n16/6/24 \nConferenciante: Pablo Hovsepian Congregación Olivos - Usted puede experimentar gozo al servir a Dios (103)\nPresidente: Roberto Schettini \nLector: Facundo Torres\nDisc. saliente: Alexis Szerman (159) Congregación Olivos - Sábados 19 horas\n23/6/24 \nConferenciante: Alejandro Palmieri Congregación Olivos - Cómo lograr que Dios escuche nuestras oraciones (86)\nPresidente: Sebastián Bertino \nLector: Alberto Romero\nDisc. saliente Roberto Schettini (142) Congregación Olivos - Sábados 19 horas\n30/6/24\nConferenciante: A confirmar -  Congregación\nPresidente: Pablo Burtin \nLector: Usvaldo Farroni\nDisc. saliente: A Confirmar - Congregación",
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "Hola! Quisiera saber cual es la conferencia para este domingo?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "¡Hola! La conferencia para este domingo 5 de mayo será dictada por Carlos Capeluto de la Congregación Villa Crespo y se titula \"¿En las promesas de quién confía usted?\" (61).  \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "No, para mañana domingo 19",
      ],
    },
    {
      "role": "model",
      "parts": [
        "¡Ah, claro! Para el domingo 19 de mayo, la conferencia será dictada por Andrés Gibezzi de la Congregación Villa Crespo y se titula \"¿Dónde encontrará una esperanza segura?\" (62). \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Y de donde es la salida de hoy sabado del grupo 3?",
        " de donde es la salida de hoy sabado del grupo 3?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "¡Hola! El grupo 3 se reúne hoy sábado a las 10:00hs en Primera Junta 2226 (Flia. Benito) o Pedernera 685 (Flia. Barrese).  \n\n¿Necesitas saber algo más? 😊\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Que asignaciones tiene alexis szerman para el miercoles 22/05?",
        "no la reunion del 20/05 me referia",
      ],
    },
    {
      "role": "model",
      "parts": [
        "¡Ah, claro! Para la reunión del 20/05, Alexis Szerman tiene las siguientes asignaciones:\n\n* **Estudio bíblico de la congregación:**  Dirigirá el estudio bíblico de la congregación.\n\nEspero que esta información te sea útil. 😊\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Puedes decirme quienes mas tienen asignaciones la reunion del 20/05?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "¡Claro que sí! Aquí tienes las asignaciones para la reunión del 20/05/2024:\n\n**Presidente:** Gustavo Gómez\n**Oración Inicial:** Pablo Burtin\n**Canción inicial:** 102\n**Tesoros de la Biblia - ¿Por qué ayudar a los demás?:** Jonatan Romano\n**Busquemos perlas escondidas:** Jorge Pedevilla\n**Lectura de la Biblia Sala A:** Reinaldo Jarro\n**Lectura de la Biblia Sala B:** Brian Pérez\n**Empiece conversaciones PREDICACIÓN INFORMAL Sala A:** Jaqueline Sinisterra Ayudante: Pilar Benito\n**Empiece conversaciones PREDICACIÓN INFORMAL Sala B:** Sandra Varela Ayudante: Anahí Pereda\n**Empiece conversaciones PREDICACIÓN INFORMAL Sala A:** Noelia Bertino Ayudante: Julieta Szerman\n**Empiece conversaciones PREDICACIÓN INFORMAL Sala B:** Agustina Farroni Ayudante: Carina Vera\n**Haga discípulos Sala A:** Anahí Romano Ayudante: Camila Fernandez\n**Haga discípulos Sala B:** Luz Gimenez Ayudante: Norma Ruíz\n**Canción intermedia 138**\n**Hagamos el bien a nuestros hermanos mayores:** Victor Vera\n**Estudio bíblico de la congregación:** Alexis Szerman\n**Lector Libro:** Sebastián Bertino\n**Canción Final 84**\n**Oracion de conclusión:** Alberto Romero\n\n¿Te gustaría saber algo más sobre la reunión? 😊\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "quien es el presidente de la reunion del 29/05?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "El presidente de la reunión del 29/05 es Pablo Burtin. \n\n¿Te gustaría saber algo más sobre la reunión del 29/05? 😊\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "De donde es la salida del proximo lunes?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "El próximo lunes 20 de mayo, la salida es a las 10:00hs en Av. Goyena Puán. La conductora es Graciela Fernández y los territorios son 53 y 54. \n\n¿Te gustaría saber algo más sobre las salidas de predicación? 😊\n\n\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "De donde será la salida el proximo jueves a la tarde?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "El próximo jueves 23 de mayo, a la tarde, la salida es a las 18:15hs en Falcón y Quirno. La conduce Gustavo Gómez y los territorios son 3 y 4. \n\n¿Te gustaría saber algo más sobre las salidas de predicación? 😊\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Y la salida del martes a la mañana?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "El martes 21 de mayo, a la mañana, la salida es a las 10:00hs en Av. Rivadavia y Gavilán. La conduce Alejandra Scanegatta y los territorios son 12 y 15. \n\n¿Te gustaría saber algo más sobre las salidas de predicación? 😊\n\n\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Luciano Sinisterra tiene alguna asignacion en mayo  y junio?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "¡Claro que sí! Luciano Sinisterra tiene una asignación en la reunión del 27 de mayo:\n\n* **Busquemos perlas escondidas:**  En la reunión del 27 de mayo, Luciano estará a cargo de la sección \"Busquemos perlas escondidas\".\n\nAdemás, en la reunión pública del 2 de junio, Luciano será el lector. \n\n¿Te gustaría saber algo más sobre las asignaciones de Luciano o sobre las reuniones? 😊\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Quien da la conferencia este domingo?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Este domingo 19 de mayo, la conferencia será dictada por Andrés Gibezzi de la Congregación Villa Crespo. \n\n¿Te gustaría saber algo más sobre la conferencia? 😊\n\n\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "cual es el tema de la conferencia?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "El tema de la conferencia de este domingo 19 de mayo es \"¿Dónde encontrará una esperanza segura?\" (62). \n\n¿Te gustaría saber algo más sobre la conferencia? 😊\n\n\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Que asignaciones tiene Alexis Szerman en mayo y junio?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "¡Claro que sí! Aquí tienes las asignaciones de Alexis Szerman para mayo y junio:\n\n**Mayo:**\n\n* **20/05:** Estudio bíblico de la congregación\n* **27/05:** Tesoros de la Biblia - Aproveche al máximo la enseñanza que viene de Jehová\n\n**Junio:**\n\n* **2/06:** Presidente de la reunión pública\n* **16/06:** Disc. saliente en la reunión pública\n\n¿Te gustaría saber algo más sobre las asignaciones de Alexis Szerman o sobre las reuniones? 😊\n\n\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "De donde se sale a predicar mañana por la mañana?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Mañana, sábado 18 de mayo, no hay salida programada para la congregación. Las salidas de los grupos son solo los sábados por la mañana. \n\n¿Te gustaría saber algo más sobre las salidas de predicación? 😊\n\n\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Quien hace la lectura del libro hoy?",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Quien hace la lectura del libro hoy?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "¡Hola! Puedes decirme que fecha es hoy?\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Hoy es 22/05. Quien tiene la lectura del libro",
      ],
    },
    {
      "role": "model",
      "parts": [
        "¡Ah, claro! Hoy, 22 de mayo, la lectura del libro la tiene Sebastián Bertino. \n\n¿Te gustaría saber algo más sobre la reunión de hoy? 😊\n\n\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Quien conduce el libro?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "El que conduce el libro es Alexis Szerman. \n\n¿Te gustaría saber algo más sobre la reunión de hoy? 😊\n\n\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "De donde se sale a predicar mañana?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "¿Qué fecha es mañana? \n\n\n\n\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Jueves 23",
      ],
    },
    {
      "role": "model",
      "parts": [
        "El jueves 23 de mayo, a la mañana, la salida es a las 10:00hs en Primera Junta y Camacuá. La conduce Pilar Benito y los territorios son 39 y 40. \n\n¿Te gustaría saber algo más sobre las salidas de predicación? 😊\n\n\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Quien da la conferencia este domingo?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Este domingo 26 de mayo, la conferencia será dictada por Alberto Ventimiglia de la Congregación Villa Crespo. \n\n¿Te gustaría saber algo más sobre la conferencia? 😊\n\n\n\n\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "quien es presidente y el lector el domingo?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "El presidente de la reunión pública este domingo 26 de mayo es Jorge Pedevilla. El lector es Lucas Herrera. \n\n¿Te gustaría sa",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Hola! de donde es la salida mañana?",
      ],
    },
    {
      "role": "model",
      "parts": [
        "¡Hola!  ¿Mañana te refieres a sábado 25 de mayo? \n\nSi es así, hay un día especial de predicación ese día. La salida es a las 10:30hs en Primera Junta 2226 (Flia. Benito) y la conduce Oscar Fioriti. Los territorios son 29, 30 y 27.\n\n¿Te gustaría saber algo más sobre la salida de mañana? 😊\n",
      ],
    },
  ]
)

response = chat_session.send_message("INSERT_INPUT_HERE")


print(response.text)
print(chat_session.history)

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("userInput")
    if not user_input:
        return jsonify({"error": "No user input provided"}), 400
    
    # Send user input to the chat model
    response = chat_session.send_message(user_input)
    
    return jsonify({"response": response.text})

if __name__ == "__main__":
    app.run(debug=True)



