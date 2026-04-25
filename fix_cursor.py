content = open('templates/base.html', encoding='utf-8').read() 
fix = '* { cursor: default; }\na, button, [onclick], label, select { cursor: pointer !important; }\ninput, textarea { cursor: text !important; }' 
content = content.replace('{% block extra_css %}{% endblock %}', fix + '\n{% block extra_css %}{% endblock %}') 
open('templates/base.html', 'w', encoding='utf-8').write(content) 
